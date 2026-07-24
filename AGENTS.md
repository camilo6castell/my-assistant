# AGENTS.md — MyAssistant RAG Multi-Context

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
│   LangGraph
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

Backend and frontend are intentionally decoupled.

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
src/graph/graph.py
src/graph/nodes.py
src/llm/providers.py
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
CLI / API

↓

Graph

↓

Retrieval

↓

LLM Providers

↓

External Backends
```

Business logic should never bypass these layers.

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

LLMs are selected per role.

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

## Graph

Routing belongs in the graph.

Business nodes should not contain routing decisions unless explicitly intended.

---

## API

The API is stateless.

Do not introduce server-side session state.

Clients send:

- collections
- chat_history

with every request.

---

## Frontend

Frontend communicates exclusively through the API.

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

Avoid modifying graph logic.

---

## Adding a new graph node

Update

```
graph.py
nodes.py
state.py
```

Keep node responsibilities small.

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
- Role configuration lives in .env.providers.
- Backend/model routing is runtime configurable.
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
