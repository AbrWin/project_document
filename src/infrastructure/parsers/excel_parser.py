from __future__ import annotations

"""
Infrastructure → Parsers → Excel Parser
Parses .xlsx / .xls files into a list of row-objects using pandas.

Each row becomes a dict: { column_name: cell_value, ... }
NaN cells → None, datetimes → ISO string, numbers → int or float.
"""

import io
import math
from datetime import date, datetime
from typing import Any, Optional

import pandas as pd


class ExcelSheetResult:
    """Parsed result for a single sheet."""

    def __init__(
        self,
        sheet_name: str,
        columns: list[str],
        rows: list[dict[str, Any]],
    ) -> None:
        self.sheet_name = sheet_name
        self.columns = columns
        self.rows = rows

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def to_text_rows(self) -> list[str]:
        """
        Convert each row to a human-readable string for vector ingestion.
        Example: "nombre: Laptop X1 | precio: 1299 | stock: 50"
        """
        return [
            " | ".join(
                f"{col}: {val}"
                for col, val in row.items()
                if val is not None and str(val).strip() != ""
            )
            for row in self.rows
        ]


def _sanitize_value(val: Any) -> Any:
    """Convert pandas/numpy types to plain Python types safe for JSON."""
    if val is None:
        return None
    # NaN / Inf → None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    # numpy int / float
    try:
        import numpy as np
        if isinstance(val, np.integer):
            return int(val)
        if isinstance(val, np.floating):
            return None if math.isnan(float(val)) else float(val)
        if isinstance(val, np.bool_):
            return bool(val)
    except ImportError:
        pass
    # datetime / date → ISO string
    if isinstance(val, (datetime, pd.Timestamp)):
        return pd.Timestamp(val).isoformat()
    if isinstance(val, date):
        return val.isoformat()
    # pandas NA
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return val


def parse_excel(
    file_bytes: bytes,
    filename: str = "file.xlsx",
    sheet_name: Optional[str] = None,
) -> list[ExcelSheetResult]:
    """
    Parse an Excel file and return a list of ExcelSheetResult (one per sheet).

    Args:
        file_bytes:  Raw bytes of the .xlsx / .xls file.
        filename:    Original filename (used to detect engine).
        sheet_name:  If given, parse only that sheet. Otherwise parse all sheets.

    Returns:
        List of ExcelSheetResult. For single-sheet requests the list has one item.

    Raises:
        ValueError: If file cannot be parsed or sheet not found.
    """
    engine = "xlrd" if filename.lower().endswith(".xls") else "openpyxl"

    try:
        excel_file = pd.ExcelFile(io.BytesIO(file_bytes), engine=engine)
    except Exception as exc:
        raise ValueError(f"Cannot read Excel file '{filename}': {exc}") from exc

    available_sheets = excel_file.sheet_names

    if sheet_name:
        if sheet_name not in available_sheets:
            raise ValueError(
                f"Sheet '{sheet_name}' not found. "
                f"Available sheets: {available_sheets}"
            )
        sheets_to_parse = [sheet_name]
    else:
        sheets_to_parse = available_sheets

    results: list[ExcelSheetResult] = []

    for name in sheets_to_parse:
        try:
            df = pd.read_excel(excel_file, sheet_name=name, engine=engine)
        except Exception as exc:
            raise ValueError(f"Error reading sheet '{name}': {exc}") from exc

        # Normalize column names → strip whitespace
        df.columns = [str(col).strip() for col in df.columns]

        # Remove fully empty rows
        df = df.dropna(how="all")

        columns = list(df.columns)
        rows: list[dict[str, Any]] = []

        for _, row in df.iterrows():
            sanitized = {col: _sanitize_value(row[col]) for col in columns}
            rows.append(sanitized)

        results.append(ExcelSheetResult(sheet_name=name, columns=columns, rows=rows))

    return results
