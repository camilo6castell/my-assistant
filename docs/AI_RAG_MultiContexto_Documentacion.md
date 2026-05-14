<div align="center">

<img src="https://img.shields.io/badge/MyAssistant-RAG_Multi--Context-0d1117?style=for-the-badge&logo=openai&logoColor=white" alt="RAG Multi-Context" height="60"/>

# MyAssistant — RAG Multi-Context

**Private semantic memory engine. Local LLM. Zero cloud dependency. Total control.**

[![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FAISS](https://img.shields.io/badge/FAISS-IndexFlatIP-blue?style=flat-square)](https://github.com/facebookresearch/faiss)
[![FlagEmbedding](https://img.shields.io/badge/FlagEmbedding-BAAI%2Fbge--small--en-orange?style=flat-square)](https://huggingface.co/BAAI/bge-small-en-v1.5)
[![Ollama](https://img.shields.io/badge/Ollama-qwen2.5:14b-black?style=flat-square)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](./LICENSE)

</div>

---

## Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Ingestion Pipeline](#-ingestion-pipeline)
- [Search & Re-ranking](#-search--re-ranking)
- [Query Modes](#-query-modes)
- [Prompt Engineering](#-prompt-engineering)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Contributing](#-contributing)
- [License](#-license)

---

## 📡 Overview

MyAssistant is a **fully local, privacy-first Retrieval-Augmented Generation (RAG) system** designed as a private semantic memory infrastructure. It allows querying multiple independent knowledge domains — philosophy, technical documentation, web content, etc. — through a local LLM that reasons exclusively from retrieved context.

Nothing leaves the machine. No API keys. No cloud embeddings. No telemetry.

The system is built around three core principles:

**Isolation** — each knowledge domain lives in its own FAISS index with independent embeddings, metadata, and vectors. Contexts never bleed into each other.

**Control** — every component of the pipeline (chunking, embedding, indexing, retrieval, re-ranking, prompt construction, LLM call) is explicit and tunable. There are no black-box abstractions.

**Honesty** — the LLM is hard-constrained by the prompt to answer only from retrieved context. It cannot hallucinate from general knowledge. When the answer is not in the documents, it says so.

---

## 🏛️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User Input (CLI)                             │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
              ┌────────────────▼──────────────────┐
              │         Context Selection         │
              │   /context <name> → load_context  │
              │   reads index.faiss + metadata.pkl│
              │         + vectors.npy             │
              └────────────────┬──────────────────┘
                               │
              ┌────────────────▼──────────────────┐
              │         Query Expansion           │
              │  build_search_query(question)     │
              │  RIGOROUS:  1 query               │
              │  INTERPRET: 3 expanded queries    │
              │  (+ conversation history context) │
              └────────────────┬──────────────────┘
                               │
              ┌────────────────▼──────────────────┐
              │         FAISS Search              │
              │  BAAI/bge-small-en-v1.5 encoding  │
              │  L2 normalization → cosine sim    │
              │  IndexFlatIP.search() → top-K     │
              │  candidate pool (union of queries)│
              └────────────────┬──────────────────┘
                               │
              ┌────────────────▼──────────────────┐
              │       Custom Re-ranking           │
              │  score each candidate against     │
              │  ALL query embeddings             │
              │  avg cosine score → sort → top-N  │
              └────────────────┬──────────────────┘
                               │
              ┌────────────────▼──────────────────┐
              │       Prompt Construction         │
              │  chat history (last N turns) +    │
              │  retrieved chunks + mode rules +  │
              │  question → structured prompt     │
              └────────────────┬──────────────────┘
                               │
              ┌────────────────▼──────────────────┐
              │         Ollama (local LLM)        │
              │   qwen2.5:14b-instruct-q4_K_M     │
              │   temp 0.2 (rigorous)             │
              │   temp 0.7 (interpretive)         │
              └────────────────┬──────────────────┘
                               │
              ┌────────────────▼──────────────────┐
              │    Response + Confidence Score    │
              │    avg cosine score displayed     │
              │    chat memory appended           │
              └───────────────────────────────────┘
```

### Filesystem layout

Each context is a self-contained directory under `/srv/ai/vectorstores/`:

```
/srv/ai/
├── vectorstores/
│   ├── filosofia/
│   │   ├── index.faiss      # FAISS IndexFlatIP — vector search index
│   │   ├── metadata.pkl     # list of {source, page, text} per chunk
│   │   └── vectors.npy      # raw embeddings matrix (used for re-ranking)
│   ├── react/
│   │   └── ...
│   └── <any-domain>/
│       └── ...
└── data/
    └── <files to ingest — PDF, HTML, TXT>
```

Contexts are fully independent: loading one never affects another.

---

## 🔄 Ingestion Pipeline

All three ingestion modes share the same pipeline: **chunk → embed → normalize → index → persist**.

```
Source (file / URL / domain)
         │
         ▼
  Text extraction
  (PdfReader / BeautifulSoup / readability-lxml)
         │
         ▼
  chunk_text(text)
  ├── chunk_size:    500 chars
  └── chunk_overlap: 100 chars  (sliding window — preserves context across boundaries)
         │
         ▼
  FlagModel.encode(chunks)       ← BAAI/bge-small-en-v1.5
  numpy float32 cast
  faiss.normalize_L2(embeddings) ← L2 norm converts inner product to cosine similarity
         │
         ▼
  Append to existing index       ← incremental — never rebuilds from scratch
  Update vectors.npy             ← full embedding matrix for re-ranking
  Update metadata.pkl            ← {source, page, text} per chunk
  Write index.faiss
```

### Ingestion modes

| Script           | Source                       | Use case                               |
| ---------------- | ---------------------------- | -------------------------------------- |
| `ingest.py`      | Local files (PDF, HTML, TXT) | Books, exported docs, downloaded pages |
| `web_ingest.py`  | Single URL                   | Specific articles, reference pages     |
| `web_crawler.py` | Full domain (up to 50 pages) | Official docs, wikis, structured sites |

All modes are **incremental**: already-indexed sources are skipped via `existing_sources` deduplication.

---

## 🔍 Search & Re-ranking

The retrieval system goes beyond a single FAISS lookup. It implements a **multi-query candidate expansion + cosine re-ranking** pipeline:

### Step 1 — Query expansion

In **interpretive mode**, a single question expands to 3 semantically diverse queries before any search happens:

```python
queries = [
    history_text + question,                          # contextualized by chat memory
    f"Explica conceptualmente: {question}",           # conceptual framing
    f"Principio general relacionado con: {question}", # principle-level framing
]
```

This broadens the candidate pool to capture chunks that may be semantically adjacent but not lexically close to the original question.

### Step 2 — FAISS search (candidate pool)

Each query is independently embedded, normalized, and searched against the FAISS index. Candidate chunk indices from all queries are **union-merged** into a single deduplicated pool.

```python
# IndexFlatIP with L2-normalized vectors = exact cosine similarity search
distances, indices = INDEX.search(emb, TOP_K_INITIAL)
all_candidate_indices.update(indices[0])
```

### Step 3 — Re-ranking by average cosine score

Each candidate chunk is scored against **all query embeddings**, and the average cosine similarity is used to rank. This penalizes chunks that score well for only one of the expanded queries and rewards those that are broadly relevant:

```python
for idx in candidate_indices:
    chunk_vector = CHUNK_VECTORS[idx]          # loaded from vectors.npy — no re-encode
    score_sum = sum(np.dot(q_emb, chunk_vector) for q_emb in query_embeddings)
    avg_score = score_sum / len(query_embeddings)
```

The top-N chunks after re-ranking are passed to prompt construction. The average score of the final selection is shown to the user as a **confidence indicator**.

---

## 🎛️ Query Modes

The system has two operating modes toggled with `/mode`:

|                     | Rigorous mode                                 | Interpretive mode                                 |
| ------------------- | --------------------------------------------- | ------------------------------------------------- |
| **Query expansion** | 1 query                                       | 3 expanded queries                                |
| **Top-K initial**   | 15                                            | 25                                                |
| **Top-K final**     | 5                                             | 7                                                 |
| **LLM temperature** | 0.2                                           | 0.7                                               |
| **Prompt rules**    | Strict: cite source or say "not in documents" | Flexible: may connect ideas, synthesize, abstract |
| **Best for**        | Factual lookup, precise references            | Philosophical analysis, conceptual synthesis      |

---

## 🧠 Prompt Engineering

The prompt is constructed programmatically with three dynamic sections:

```
System identity: "Answer EXCLUSIVELY from provided context."

Recent chat history (last MAX_TURNS=4 turns)

Mode rules block:
  RIGOROUS:     cite source + page / infer with caveat / "not in documents"
  INTERPRETIVE: connect ideas / abstract principles / always cite source base

Retrieved context:
  "Fuente: {source} (página {page})\n{chunk_text}"
  (top-N chunks, separated by ---)

Question
```

The **rules block** is mode-specific and hardcoded in natural language to constrain the LLM's behavior. In rigorous mode, the exact string `"No está en los documentos proporcionados, ni se puede inferir de ellos."` is instructed as the mandatory response when no answer can be found — no paraphrasing allowed.

---

## 🛠️ Tech Stack

| Component            | Technology                                    |
| -------------------- | --------------------------------------------- |
| Language             | Python 3.11                                   |
| Embedding model      | `BAAI/bge-small-en-v1.5` via FlagEmbedding    |
| Vector index         | FAISS `IndexFlatIP` (exact cosine similarity) |
| Vector normalization | L2 via `faiss.normalize_L2`                   |
| LLM runtime          | Ollama (local)                                |
| LLM model            | `qwen2.5:14b-instruct-q4_K_M`                 |
| PDF parsing          | pypdf                                         |
| HTML extraction      | BeautifulSoup4 + readability-lxml             |
| Numerical ops        | NumPy                                         |
| Persistence          | `.faiss` + `.pkl` + `.npy` (no database)      |

---

## 📁 Project Structure

```
.
├── requeriments.txt
└── rag/
    ├── ingest.py        # Local file ingestion (PDF / HTML / TXT)
    ├── web_ingest.py    # Single URL ingestion
    ├── web_crawler.py   # Domain crawler (up to 50 pages, respects domain boundary)
    └── query.py         # Interactive CLI — context loading, search, re-rank, LLM call
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) installed and running locally
- At least one Ollama model pulled (default: `qwen2.5:14b-instruct-q4_K_M`)

### Install dependencies

```bash
git clone https://github.com/your-username/my-assistant.git
cd my-assistant

pip install -r requeriments.txt
```

### Pull the LLM model

```bash
ollama pull qwen2.5:14b-instruct-q4_K_M
```

> Any instruction-tuned model available in Ollama works. Smaller models (`llama3.2:3b`, `mistral:7b`) run on less VRAM. Update `MODEL_NAME` in `query.py` accordingly.

### Create the base directories

```bash
mkdir -p /srv/ai/vectorstores
mkdir -p /srv/ai/data
```

---

## 💻 Usage

### 1. Ingest local documents

Place PDF, HTML, or TXT files in `/srv/ai/data/`, then:

```bash
python rag/ingest.py <context-name>

# Example
python rag/ingest.py filosofia
```

### 2. Ingest a single URL

```bash
python rag/web_ingest.py <context-name> <url>

# Example
python rag/web_ingest.py react https://react.dev/reference/react/useState
```

### 3. Crawl an entire domain

```bash
python rag/web_crawler.py <context-name> <base-url>

# Example — crawls up to 50 internal pages
python rag/web_crawler.py react https://react.dev/reference
```

### 4. Query

```bash
python rag/query.py
```

**CLI commands:**

| Command           | Action                                        |
| ----------------- | --------------------------------------------- |
| `/context <name>` | Load a context (FAISS index) into memory      |
| `/list`           | List all available contexts                   |
| `/mode`           | Toggle between RIGOROUS and INTERPRETIVE mode |
| `/reset`          | Clear conversation memory                     |
| `/exit` or `/bye` | Exit                                          |

---

## ⚙️ Configuration

All configuration lives as module-level constants at the top of each script:

| Constant             | File             | Default                               | Description                                  |
| -------------------- | ---------------- | ------------------------------------- | -------------------------------------------- |
| `BASE_VECTOR_PATH`   | all              | `/srv/ai/vectorstores`                | Root directory for all context indices       |
| `DATA_PATH`          | `ingest.py`      | `/srv/ai/data`                        | Source directory for local file ingestion    |
| `EMBED_MODEL`        | all              | `BAAI/bge-small-en-v1.5`              | HuggingFace embedding model                  |
| `OLLAMA_URL`         | `query.py`       | `http://localhost:11434/api/generate` | Ollama API endpoint                          |
| `MODEL_NAME`         | `query.py`       | `qwen2.5:14b-instruct-q4_K_M`         | Ollama model to use                          |
| `CHUNK_SIZE`         | all              | `500`                                 | Characters per chunk                         |
| `CHUNK_OVERLAP`      | all              | `100`                                 | Overlap between consecutive chunks           |
| `BASE_TOP_K_INITIAL` | `query.py`       | `15`                                  | Candidates per query before re-ranking       |
| `BASE_TOP_K_FINAL`   | `query.py`       | `5`                                   | Chunks passed to the prompt after re-ranking |
| `MAX_TURNS`          | `query.py`       | `4`                                   | Conversation turns kept in memory            |
| `MAX_PAGES`          | `web_crawler.py` | `50`                                  | Max pages per crawl session                  |

---

## 🤝 Contributing

Contributions are welcome. Please open an issue before submitting a pull request to discuss the proposed change.

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Commit using [Conventional Commits](https://www.conventionalcommits.org/): `git commit -m 'feat: add your feature'`
4. Push and open a pull request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.

Copyright © 2025 Camilo Andres Castellanos Herrera

---

<div align="center">

_Your knowledge. Your hardware. Your rules._

</div>
