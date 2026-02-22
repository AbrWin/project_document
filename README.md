# ProyectIA Backend

Python backend built with **Clean Architecture (Hexagonal)** for AI/LLM integration via Azure OpenAI, Azure AI Inference, and direct OpenAI. Supports conversational chat, SSE streaming, RAG with pgvector, and conversation history management.

---

## Table of Contents

- [Architecture](#architecture)
- [Diagrams](#diagrams)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Database](#database)
- [Running the Server](#running-the-server)
- [Endpoints](#endpoints)
- [AI Providers](#ai-providers)
- [Project Structure](#project-structure)
- [Tests](#tests)
- [Docker](#docker)

---

## Architecture

The project is organized into **4 layers** following the Dependency Inversion principle (Ports & Adapters):

```
┌─────────────────────────────────────────────┐
│              API Layer (FastAPI)             │  ← HTTP / SSE / Schemas
├─────────────────────────────────────────────┤
│         Core Layer (Use Cases + Ports)       │  ← Business logic
├─────────────────────────────────────────────┤
│     Infrastructure Layer (Adapters)          │  ← Azure, OpenAI, PostgreSQL
├─────────────────────────────────────────────┤
│           Domain Layer (Entities)            │  ← Pure entities, no external deps
└─────────────────────────────────────────────┘
```

- **Domain**: Entities (`Message`, `Conversation`, `Document`), value objects (`LLMConfig`, `AIProvider`), and domain exceptions. No external dependencies.
- **Core**: Ports (ABC interfaces) and use cases (`ChatUseCase`, `RAGUseCase`).
- **Infrastructure**: Concrete adapters — Azure OpenAI, Azure AI Inference, direct OpenAI, SQLAlchemy + asyncpg, pgvector.
- **API**: FastAPI endpoints, Pydantic schemas, middleware, SSE streaming.

---

## Diagrams

### Architecture UML — Class Diagram

Shows all layers, their classes, interfaces (ports), concrete adapters (infrastructure), and how they relate to each other.

```mermaid
classDiagram
    namespace Domain {
        class Message {
            +id: str
            +role: MessageRole
            +content: str
            +created_at: datetime
        }
        class Conversation {
            +id: str
            +title: str
            +messages: List~Message~
            +add_message(msg)
            +get_last_n_messages(n)
        }
        class Document {
            +id: str
            +content: str
            +metadata: dict
        }
        class LLMConfig {
            +provider: AIProvider
            +temperature: float
            +max_tokens: int
            +streaming: bool
        }
        class AIProvider {
            <<enumeration>>
            AZURE_OPENAI
            AZURE_INFERENCE
            OPENAI
        }
    }
    namespace Core {
        class AIProviderPort {
            <<interface>>
            +chat(messages, config) str
            +stream(messages, config) AsyncGenerator
            +embed(text) List~float~
        }
        class ConversationRepositoryPort {
            <<interface>>
            +save(conversation) Conversation
            +find_by_id(id) Conversation
            +delete(id) void
        }
        class VectorStorePort {
            <<interface>>
            +add_documents(docs) void
            +similarity_search(query, k) List~Document~
        }
        class ChatUseCase {
            -ai_provider: AIProviderPort
            -conversation_repo: ConversationRepositoryPort
            -vector_store: VectorStorePort
            +execute(conv_id, message, use_rag) Message
            +stream(conv_id, message) AsyncGenerator
        }
        class RAGUseCase {
            -ai_provider: AIProviderPort
            -vector_store: VectorStorePort
            +ingest_document(document) void
            +search(query, k) List~Document~
        }
    }
    namespace Infrastructure {
        class AzureOpenAIProvider {
            +chat(messages, config) str
            +stream(messages, config) AsyncGenerator
            +embed(text) List~float~
        }
        class AzureInferenceProvider {
            +chat(messages, config) str
            +stream(messages, config) AsyncGenerator
            +embed(text) List~float~
        }
        class OpenAIProvider {
            +chat(messages, config) str
            +stream(messages, config) AsyncGenerator
            +embed(text) List~float~
        }
        class ConversationRepository {
            -db_session: AsyncSession
            +save(conversation) Conversation
            +find_by_id(id) Conversation
        }
        class PgVectorRepository {
            -engine: AsyncEngine
            +add_documents(docs) void
            +similarity_search(query, k) List~Document~
        }
    }
    AIProviderPort <|.. AzureOpenAIProvider : implements
    AIProviderPort <|.. AzureInferenceProvider : implements
    AIProviderPort <|.. OpenAIProvider : implements
    ConversationRepositoryPort <|.. ConversationRepository : implements
    VectorStorePort <|.. PgVectorRepository : implements
    ChatUseCase --> AIProviderPort : uses
    ChatUseCase --> ConversationRepositoryPort : uses
    ChatUseCase --> VectorStorePort : uses
    RAGUseCase --> AIProviderPort : uses
    RAGUseCase --> VectorStorePort : uses
    Conversation "1" *-- "many" Message : contains
```

---

### Chat Use Case — Flow Diagram

End-to-end flow of a chat message, from the client request through optional RAG context injection, AI provider selection, and response (blocking or streaming SSE).

```mermaid
flowchart TD
    A([Client Request]) --> B[POST /conversations/id/messages]
    B --> C{Streaming?}
    C -->|No| D[ChatUseCase.execute]
    C -->|Yes| E[ChatUseCase.stream]

    D --> F{use_rag?}
    E --> F

    F -->|Yes| G[Generate query embedding\nAIProvider.embed]
    F -->|No| H[Load conversation history\nfrom PostgreSQL]

    G --> I[VectorStore.similarity_search\npgvector cosine distance]
    I --> J[Inject retrieved context\ninto system prompt]
    J --> H

    H --> K[Build LangChain messages\nwith history + context]
    K --> L{Provider?}

    L -->|azure_openai| M[AzureOpenAIProvider]
    L -->|azure_inference| N[AzureInferenceProvider]
    L -->|openai| O[OpenAIProvider]

    M --> P{Streaming?}
    N --> P
    O --> P

    P -->|Yes| Q[Yield SSE chunks\ntext/event-stream]
    P -->|No| R[Return full JSON response]

    Q --> S[Save assistant message\nto PostgreSQL]
    R --> S
    S --> T([Response to Client])
```

---

### RAG Use Case — Flow Diagram

Two independent sub-flows: **document ingestion** (splits, embeds, and stores documents) and **semantic search** (embeds the query and retrieves the most relevant chunks).

```mermaid
flowchart TD
    subgraph INGEST["Document Ingestion Flow"]
        A1([POST /documents]) --> B1[RAGUseCase.ingest_document]
        B1 --> C1[Split text into chunks\nRecursiveCharacterTextSplitter]
        C1 --> D1[Generate embeddings\nAIProvider.embed per chunk]
        D1 --> E1[Store vectors + metadata\nin pgvector]
        E1 --> F1([200 OK])
    end

    subgraph SEARCH["Semantic Search Flow"]
        A2([GET /search?q=query]) --> B2[RAGUseCase.search]
        B2 --> C2[Generate query embedding\nAIProvider.embed]
        C2 --> D2[pgvector cosine similarity\nTOP-K results]
        D2 --> E2[Rank and filter\nby score threshold]
        E2 --> F2([Return ranked DocumentChunks])
    end
```

---

### Full Request Lifecycle — Sequence Diagram

Shows the interaction between all components for a single chat request, including authentication, optional RAG retrieval, AI streaming, and persistence.

```mermaid
sequenceDiagram
    actor Client
    participant API as FastAPI API
    participant MW as Auth Middleware
    participant UC as ChatUseCase
    participant VS as VectorStore
    participant AI as AI Provider
    participant DB as PostgreSQL

    Client->>API: POST /conversations/{id}/messages
    API->>MW: Validate X-API-Key
    MW-->>API: Authorized
    API->>UC: execute(conv_id, message, use_rag)

    UC->>DB: Load conversation history
    DB-->>UC: Conversation + Messages

    alt use_rag = true
        UC->>AI: embed(user_message)
        AI-->>UC: query_vector [1536 dims]
        UC->>VS: similarity_search(query_vector, k=5)
        VS-->>UC: relevant_chunks[]
        UC->>UC: Inject context into system prompt
    end

    UC->>AI: chat(messages, llm_config)

    alt streaming = true
        loop SSE Chunks
            AI-->>UC: token_chunk
            UC-->>API: yield chunk
            API-->>Client: data: {"content": "..."}
        end
    else streaming = false
        AI-->>UC: full_response
        UC-->>API: Message
        API-->>Client: 200 OK {"message": {...}}
    end

    UC->>DB: Save assistant message
    DB-->>UC: Saved
```

---

## Tech Stack

| Category | Technology |
|----------|------------|
| Framework | FastAPI 0.115 + Uvicorn |
| AI / LLM | LangChain 0.3.9, LangGraph 0.2 |
| Azure AI | azure-ai-inference 1.0.0b7 + azure-identity |
| OpenAI | openai 1.57, langchain-openai |
| Database | PostgreSQL 17 + pgvector |
| ORM / Async | SQLAlchemy 2.0 asyncio + asyncpg |
| Migrations | Alembic 1.14 |
| Vector Store | pgvector + langchain-postgres |
| Streaming | sse-starlette (SSE) |
| Config | Pydantic Settings v2 |
| Python | 3.9+ |

---

## Prerequisites

- Python 3.9 or higher
- PostgreSQL 17 with the `pgvector` extension
- An active Azure OpenAI, Azure AI Studio **or** OpenAI account

### Install PostgreSQL 17 + pgvector (macOS)

```bash
brew install postgresql@17
brew install pgvector   # already included in @17

brew services start postgresql@17
```

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd proyectIA

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Configuration

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

Key variables in `.env`:

```dotenv
# Default provider: azure_openai | azure_inference | openai
DEFAULT_AI_PROVIDER=azure_openai

# Azure OpenAI
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large

# Azure AI Inference (Phi-3, Mistral, Llama)
AZURE_AI_INFERENCE_ENDPOINT=https://your-model.inference.ai.azure.com
AZURE_AI_INFERENCE_KEY=your_key_here

# OpenAI Direct
OPENAI_API_KEY=your_key_here

# Database
DATABASE_URL=postgresql+asyncpg://postgres@localhost:5432/proyectia

# API Key for authentication (minimum 32 characters)
SECRET_KEY=your_very_long_secret_key_minimum_32_chars
```

---

## Database

### Create the database and enable pgvector

```bash
psql postgres -c "CREATE DATABASE proyectia;"
psql proyectia -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

> If the `postgres` role does not exist (macOS Homebrew):
> ```bash
> psql postgres -c "CREATE ROLE postgres WITH SUPERUSER LOGIN;"
> ```

### Run migrations with Alembic

```bash
# Generate initial migration
alembic revision --autogenerate -m "initial"

# Apply migration
alembic upgrade head
```

---

## Running the Server

```bash
source .venv/bin/activate
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

Available at:
- **API**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Endpoints

### General

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Server status |
| `GET` | `/health` | Detailed health check |

### Chat (`/api/v1/chat`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/conversations` | Create a new conversation |
| `GET` | `/conversations/{id}` | Get conversation with history |
| `POST` | `/conversations/{id}/messages` | Send message (full response) |
| `POST` | `/conversations/{id}/messages/stream` | Send message with SSE streaming |

### RAG (`/api/v1/rag`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/documents` | Ingest a document into the vector store |
| `GET` | `/search` | Semantic search across documents |

#### Example — Create conversation and send message

```bash
# Create conversation
curl -X POST http://localhost:8000/api/v1/chat/conversations \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{"title": "My conversation"}'

# Send message
curl -X POST http://localhost:8000/api/v1/chat/conversations/{id}/messages \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{"content": "What is LangChain?", "use_rag": false}'

# SSE Streaming
curl -N -X POST http://localhost:8000/api/v1/chat/conversations/{id}/messages/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{"content": "Explain RAG step by step"}'
```

---

## AI Providers

The backend supports 3 providers, selectable per conversation or via the default in `.env`:

| Provider | Variable | Models |
|----------|----------|--------|
| `azure_openai` | `DEFAULT_AI_PROVIDER` | GPT-4o, GPT-4, GPT-3.5 |
| `azure_inference` | `DEFAULT_AI_PROVIDER` | Phi-3, Mistral-Large, Llama-3 |
| `openai` | `DEFAULT_AI_PROVIDER` | GPT-4o, GPT-4 |

---

## Project Structure

```
proyectIA/
├── src/
│   ├── main.py                          # FastAPI entry point (app factory)
│   ├── domain/
│   │   ├── entities/
│   │   │   ├── message.py               # Message, Conversation, MessageRole
│   │   │   └── document.py              # Document, DocumentChunk
│   │   ├── value_objects/
│   │   │   └── llm_config.py            # LLMConfig, RAGConfig, AIProvider
│   │   └── exceptions/
│   │       └── ai_exceptions.py         # Domain exception hierarchy
│   ├── core/
│   │   ├── interfaces/
│   │   │   ├── ai_provider.py           # AIProviderPort (ABC)
│   │   │   └── repositories.py          # ConversationRepositoryPort, VectorStorePort
│   │   └── use_cases/
│   │       ├── chat_use_case.py         # Chat + RAG injection + streaming
│   │       └── rag_use_case.py          # Document ingestion + semantic search
│   ├── infrastructure/
│   │   ├── config/
│   │   │   └── settings.py              # Pydantic Settings v2 + lru_cache
│   │   ├── ai/
│   │   │   ├── providers/
│   │   │   │   ├── azure_openai_provider.py
│   │   │   │   ├── azure_inference_provider.py
│   │   │   │   └── openai_provider.py
│   │   │   └── provider_factory.py      # Factory pattern
│   │   ├── db/
│   │   │   ├── database.py              # Async SQLAlchemy engine
│   │   │   ├── models/
│   │   │   │   └── conversation_model.py
│   │   │   └── repositories/
│   │   │       ├── conversation_repository.py
│   │   │       └── pgvector_repository.py
│   │   └── container.py                 # Dependency injection (FastAPI Depends)
│   └── api/
│       ├── v1/
│       │   ├── schemas.py               # Pydantic schemas (Request/Response)
│       │   ├── router.py
│       │   └── endpoints/
│       │       ├── chat.py
│       │       ├── rag.py
│       │       └── health.py
│       └── middleware/
│           └── auth.py                  # API Key middleware
├── alembic/                             # Database migrations
├── tests/                               # Unit and integration tests
├── .env                                 # Environment variables (do NOT commit)
├── .env.example                         # Variables template
├── requirements.txt
├── pyproject.toml                       # ruff, black, mypy, pytest config
├── Dockerfile
└── docker-compose.yml
```

---

## Tests

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Integration tests only
pytest tests/integration/ -v
```

---

## Docker

```bash
# Start everything (API + PostgreSQL + pgvector)
docker-compose up --build

# Database only
docker-compose up db -d
```

The image includes PostgreSQL 17 with pgvector pre-configured.

---

## Compatibility Notes

- **Python 3.9**: All files include `from __future__ import annotations` to support the `X | Y` union syntax. SQLAlchemy models use `Optional[str]` and `List[...]` from `typing` instead of the newer syntax, as SQLAlchemy re-evaluates annotations internally.
- **pgvector**: Requires PostgreSQL 17 (on macOS with Homebrew, `pgvector` is only available for `postgresql@17` and `postgresql@18`).
