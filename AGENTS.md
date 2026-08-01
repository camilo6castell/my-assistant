# AGENTS.md — Ragsody RAG Multi-Context

> High-signal instructions for AI coding agents (OpenCode, Claude Code, Codex CLI, Gemini CLI, Cursor, Continue).
> This file describes **how to work in this repository**, not how to use the application.

---

# Development Philosophy

When modifying this repository, prioritize:

1. Correctness
2. Simplicity
3. Consistency
4. Reuse
5. Performance

Prefer extending existing abstractions over creating new ones.

Avoid unnecessary architectural changes.

When in doubt, preserve the existing design.

---

# Project Overview

This repository contains two independent projects sharing the same repository.

```
repo/
│
├── src/
│   Python 3.11
│   LangGraph (legacy, see below)
│   FastAPI
│   FAISS
│   Multi-provider LLM pipeline
│
└── ui/
    React 19
    TypeScript
    Vite
    Tailwind v4
    shadcn/ui
```

Backend and frontend are intentionally decoupled — each has its own model configuration.

**Model configuration is fully independent:**
- **Backend** reads from `.env.providers` + `src/config/models/models.json` for its own models. Used by `POST /api/v1/query`.
- **Frontend** reads from `.env` + `ui/src/config/models/models.json` for its own models. Used by the in-browser LangGraph agent.

The **production backend** runs two services:
- **MCP server** (`python -m src.main mcp`) — Streamable HTTP MCP server exposing `list_collections` and `retrieve_chunks` tools for AI clients.
- **REST API** (`python -m src.main api`) — FastAPI with `POST /api/v1/query` (linear RAG), `POST /api/v1/demo/query` (streaming demo), file upload, and provider config.

The LangGraph.js agent port lives in `ui/src/graph/` and runs entirely in the browser (see Fases 7–9). The original Python LangGraph agent was removed from the backend in Fase 3 and is no longer present in the codebase. The frontend agent uses its own model config (`ui/src/config/models/`) and retrieves context via MCP.

---

# Repository Layout

```
src/
    api/
    chat/
    cli/
    config/
    context/
    graph/
    ingest/
    llm/
    mcp_server/
    prompts/
    retrieval/
    utils/

ui/
    src/
```

---

# First Files To Read

Before making significant changes, read these files first.

Backend

```
src/config/settings.py
src/retrieval/search.py
src/llm/providers.py
src/mcp_server/server.py
src/api/routers/demo.py
src/main.py
```

Frontend

```
ui/vite.config.ts
ui/src/main.tsx
```

Configuration

```
.env.example
README.md
```

---

# Development Workflow

When implementing a change:

1. Understand the existing implementation.
2. Search for similar code before creating new abstractions.
3. Reuse existing helpers whenever possible.
4. Keep edits localized.
5. Avoid unrelated refactoring.
6. Preserve backward compatibility whenever practical.

Large architectural changes should only happen when clearly justified.

---

# Architecture

The project follows a layered architecture.

```
CLI / API / MCP

↓

Retrieval

↓

LLM Providers

↓

External Backends
```

Business logic should never bypass these layers.

The LangGraph.js agent (`ui/src/graph/`) runs entirely in the browser and replaced the original Python LangGraph agent (removed in Fase 3).

---

# Core Invariants

These rules should never be violated.

## Configuration

Configuration comes from

- .env
- .env.providers

loaded through

```
src/config/settings.py
```

Never hardcode configuration values.

---

## Provider Model

LLMs are selected per role. Each project has its OWN provider model:

**Backend** (`src/config/models/`):
- Reads from `.env.providers` + `src/config/models/models.json`
- Used by `POST /api/v1/query`

**Frontend** (`ui/src/config/models/`):
- Reads from `.env` + `ui/src/config/models/models.json`
- Used by the in-browser LangGraph agent

Both follow the same architecture:

Role

↓

backend,model

↓

Provider

↓

Client

Never hardcode model names inside business logic.

---

## Retrieval

Retrieval is responsible for obtaining context.

Generation should never directly query vectorstores.

---

## API

The API is stateless.

Do not introduce server-side session state.

Clients send:

- collections
- chat_history

with every request.

Note: `POST /api/v1/query/agent` (the LangGraph agent endpoint) was removed in Fase 3. Only the linear pipeline `/query` remains.

---

## Frontend

Frontend communicates through the API for simple queries (`POST /api/v1/query`)
and through MCP for retrieval when running the in-browser agent.

**Two modes of operation:**
- **`backend` mode** (default): sends `POST /api/v1/query` to the backend. The backend
  uses its own model config (`.env.providers`). Simple linear pipeline.
- **`client_agent` mode** (opt-in via Agent toggle): runs the full LangGraph.js pipeline
  in the browser (retrieve via MCP → evaluate → reformulate → generate → review → correct).
  Uses the **frontend's own model config** (`.env` + `ui/src/config/models/models.json`).

**Frontend model configuration lives in:**
- `.env` — role-to-backend+model mapping (`LLM_ROL_GENERATE`, `LLM_ROL_SUPPLEMENT`, etc.)
- `ui/src/config/models/models.json` — per-model capabilities (context_window, temperature, etc.)
- `ui/src/config/models/` — TypeScript modules mirroring the Python backend architecture

The frontend does NOT depend on `GET /api/v1/config/providers` for model capabilities.
It resolves everything from its own `models.json`.

Avoid duplicating backend logic inside React components.

---

# Extension Guide

## Adding a new LLM model

Only update

```
src/config/models/
```

No graph modifications should be necessary.

---

## Adding a new provider

Update

```
src/config/models/

src/llm/providers.py
```

Create backend implementation if required.

No graph modifications should be necessary (the graph is archived).

---

## Adding a REST endpoint

Place business logic outside FastAPI routes.

Routes should orchestrate.

Services should implement behavior.

---

## Adding configuration

All new configuration must

- have defaults
- be validated
- be documented in

```
.env.example
```

Never read environment variables directly.

Always use Settings.

---

# Vector Stores

Vector stores are located under

```
AI_HOME/vector_stores/
```

Embedder changes create a different storage path.

Changing the embedder intentionally invalidates previous indexes.

Do not migrate indexes automatically.

---

# Frontend

Technology

- React 19
- TypeScript
- Vite
- Tailwind CSS v4
- shadcn/ui
- TanStack Query
- Zustand

Alias

```
@
```

maps to

```
ui/src
```

---

# Code Style

Python

- strict mypy
- Ruff
- 100 character lines
- type annotations everywhere
- double quotes

TypeScript

- strict typing
- avoid any
- functional React components
- hooks over classes

General

Prefer readability over cleverness.

---

# Design Principles

Prefer

- composition
- dependency injection
- pure functions
- immutable data

Avoid

- duplicated logic
- global mutable state
- hidden side effects
- unnecessary abstractions

---

# Error Handling

Fail early.

Validate inputs.

Raise meaningful exceptions.

Never silently ignore errors.

Never swallow exceptions without logging.

---

# Security

Never

- commit API keys
- commit .env
- log secrets
- expose provider credentials
- hardcode tokens

---

# Performance

Avoid

- unnecessary LLM calls
- duplicate embeddings
- repeated vector searches

Reuse cached clients whenever possible.

---

# Common Pitfalls

- Always execute backend commands from repository root.
- Do not use pip install -e.
- Embedder changes invalidate indexes.
- Backend role configuration lives in `.env.providers`.
- Frontend role configuration lives in `.env` (NOT `.env.providers`).
- Backend/model routing is runtime configurable.
- Frontend and backend have **independent** `models.json` files: `src/config/models/models.json` (backend) and `ui/src/config/models/models.json` (frontend). Keep them in sync when adding models.
- The `ui/src/config/models/` directory mirrors `server/src/config/models/` in TypeScript.
- The web command starts both FastAPI and Vite.
- pnpm is preferred.

---

# Validation Checklist

Before considering a task complete:

Backend

- ruff check .
- ruff format --check .
- mypy .
- pytest -q

Frontend

- pnpm lint
- pnpm build

Verify

- imports
- typing
- formatting
- no duplicated logic
- no dead code

---

# Useful Commands

Backend

```bash
python -m src.main chat
python -m src.main api
python -m src.main mcp
python -m src.main web

python -m src.main ingest <category> <collection>

python -m src.main ingest-url <category> <collection> <url>

python -m src.main crawl <category> <collection> <url>
```

Quality

```bash
ruff check .

ruff format --check .

mypy .

pytest -q
```

Frontend

```bash
cd ui

pnpm install

pnpm run dev

pnpm run build

pnpm run lint
```

---

# Agent Guidelines

Before changing code

✓ Read existing implementation.

✓ Search for similar code.

✓ Reuse existing abstractions.

During implementation

✓ Keep changes minimal.

✓ Preserve architecture.

✓ Prefer consistency over novelty.

Before finishing

✓ Run quality checks.

✓ Remove unused code.

✓ Verify imports.

✓ Ensure documentation stays accurate.

The best change is usually the smallest change that fully solves the problem.
