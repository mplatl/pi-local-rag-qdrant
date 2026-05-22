# pi-local-rag-qdrant

> Fork of [pi-local-rag](https://github.com/vahidkowsari/pi-local-rag) backed by [Qdrant](https://qdrant.tech/) — adding **first-class collection/namespace support** for separate per-project indexes.

Local hybrid BM25 + vector RAG pipeline for the [Pi coding agent](https://github.com/badlogic/pi-mono). Index your local files into Qdrant collections and search them with keyword + vector matching.

## Key Differences from pi-local-rag

| Feature | pi-local-rag | pi-local-rag-qdrant |
|---|---|---|
| **Backend** | JSON files (`~/.pi/rag/`) | Qdrant vector database |
| **Collections** | ❌ Single flat index | ✅ Multiple named collections |
| **Storage** | Flat JSON on disk | Qdrant manages persistence |
| **Search** | In-memory scan + BM25 | Qdrant vector search + BM25 re-rank |
| **Dependencies** | `@xenova/transformers` | `@xenova/transformers` + `@qdrant/js-client-rest` + Qdrant |

## Features

- **Collections** — organize separate indexes per project (`/rag collection create my-project`)
- **Hybrid BM25 + vector search** — Qdrant vector similarity + client-side BM25 re-ranking
- **Smart chunking** — splits files into ~50-line blocks at natural blank-line boundaries
- **Incremental indexing** — skips unchanged files (SHA-256 hash check)
- **Zero cloud dependency** — uses only local Qdrant + Transformers.js (ONNX)
- **3 AI tools** — `rag_index`, `rag_query`, `rag_status` for the agent to use directly

## Prerequisites

You need a running Qdrant instance:

```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

Or point to a remote instance via `PI_QDRANT_URL`:

```bash
export PI_QDRANT_URL="http://my-server:6333"
```

## Install

```bash
pi install git:github.com/vahidkowsari/pi-local-rag-qdrant
```

## Commands

| Command | Description |
|---|---|
| `/rag index <path> [--collection <name>]` | Index a file or directory into a collection |
| `/rag search <query> [--collection <name>]` | Search indexed content |
| `/rag status [--collection <name>]` | Show index stats, all collections or one |
| `/rag rebuild [--collection <name>]` | Re-index changed files, prune deleted |
| `/rag clear [--collection <name>]` | Clear a collection or all |
| `/rag on` | Enable auto-injection |
| `/rag off` | Disable auto-injection |
| `/rag ext list` | List active indexable file extensions |
| `/rag ext add <.ext>` | Add an extension (e.g. `.cs`, `.tex`, `.zig`) |
| `/rag ext remove <.ext>` | Stop indexing files with this extension |
| `/rag ext reset` | Restore the default extension list |
| `/rag collection list` | List all collections |
| `/rag collection create <name>` | Create a new collection |
| `/rag collection delete <name>` | Delete a collection |
| `/rag collection use <name>` | Set a collection as the default |

## Example Session

```text
# Create collections for different projects
$ /rag collection create frontend-app
✅ Collection "frontend-app" created.

$ /rag collection create backend-api
✅ Collection "backend-api" created.

$ /rag collection use frontend-app
✅ Default collection set to "frontend-app".

# Index each project into its own collection
$ /rag index ~/code/frontend --collection frontend-app
Found 312 files to index → collection "frontend-app"
Indexing  ████████████████████████  100%
✅ Indexed 312 files (1,450 chunks) → "frontend-app" · 0 unchanged · 32.1s

$ /rag index ~/code/backend --collection backend-api
Found 189 files to index → collection "backend-api"
Indexing  ████████████████████████  100%
✅ Indexed 189 files (920 chunks) → "backend-api" · 0 unchanged · 18.7s

# Search within a specific collection
$ /rag search "stripe webhook" --collection backend-api
🔍 3 results for "stripe webhook" in "backend-api"  hybrid BM25+vector

webhooks.rs:45-92  bm25=0.85 vec=0.91 hybrid=0.88
  pub async fn verify_stripe_webhook(req: Request) -> Result<...> {
    let sig = req.headers().get("stripe-signature");

# Global status
$ /rag status
🔍 pi-local-rag-qdrant

  Qdrant URL:       http://127.0.0.1:6333
  Collections:      2
    (2 with data)
  Files indexed:    501
  Total vectors:    2,370
  Embedding model:  Xenova/all-MiniLM-L6-v2
  Storage:           /home/you/.pi/rag-qdrant (+ Qdrant)

  RAG injection:    enabled  topK=5  threshold=0.1  alpha=0.4
  Default collection: frontend-app

  Collections:
    backend-api     189 files · 920 vectors
    frontend-app    312 files · 1,450 vectors ◀
```

> Auto-injection uses the **default collection**. Change it with `/rag collection use <name>`.

## AI Tools

The extension registers three tools the agent can call directly:

- **`rag_index`** — Index a path into a collection (with optional `collection` param)
- **`rag_query`** — Hybrid BM25+vector search in a collection
- **`rag_status`** — Show collection stats and RAG config

## How It Works

1. **Index** — files are chunked (~50 lines each), embedded with `Xenova/all-MiniLM-L6-v2` (384-dim), and upserted into Qdrant
2. **Search** — Qdrant vector search finds candidates, then BM25 re-ranks: `alpha × BM25 + (1-alpha) × cosine_similarity` (default `alpha=0.4`)
3. **Auto-inject** — before every agent turn, the prompt is searched in the default collection and relevant chunks are prepended to the system prompt

## Storage

- **Qdrant** stores all vectors and payloads (file content, metadata)
- **`~/.pi/rag-qdrant/`** stores:
  - `config.json` — RAG settings (topK, alpha, extensions, default collection)
  - `collections.json` — file hash metadata for fast skip-on-unchanged checks

## Configuration

| Setting | Default | Description |
|---|---|---|
| `PI_QDRANT_URL` | `http://127.0.0.1:6333` | Qdrant connection URL |
| `ragEnabled` | `true` | Auto-inject context before each turn |
| `ragTopK` | `5` | Max chunks to inject |
| `ragScoreThreshold` | `0.1` | Min hybrid score to include |
| `ragAlpha` | `0.4` | BM25/vector blend (0=pure vector, 1=pure BM25) |
| `extraExtensions` | `[]` | Extra file extensions to index beyond the defaults |
| `excludeExtensions` | `[]` | Default extensions to skip |
| `defaultCollection` | `"default"` | Collection used for auto-injection and default operations |

## Migrating from pi-local-rag

If you're already using `pi-local-rag`, you'll need to re-index:

```bash
pi remove pi-local-rag
pi install git:github.com/vahidkowsari/pi-local-rag-qdrant
# Then re-index your directories into collections
docker run -d -p 6333:6333 qdrant/qdrant
pi  # restart pi
/rag collection create my-code
/rag index ~/code --collection my-code
```
