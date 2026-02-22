from __future__ import annotations

"""Domain → Entities → Message
Core business entity. No framework dependencies.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    role: MessageRole
    content: str
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.utcnow)
    conversation_id: Optional[UUID] = None
    metadata: dict = field(default_factory=dict)

    def to_langchain_dict(self) -> dict:
        return {"role": self.role.value, "content": self.content}


@dataclass
class Conversation:
    id: UUID = field(default_factory=uuid4)
    title: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    messages: list[Message] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add_message(self, message: Message) -> None:
        message.conversation_id = self.id
        self.messages.append(message)
        self.updated_at = datetime.utcnow()

    def get_history(self) -> list[dict]:
        return [m.to_langchain_dict() for m in self.messages]

    def get_last_n_messages(self, n: int) -> list[Message]:
        return self.messages[-n:] if len(self.messages) >= n else self.messages
