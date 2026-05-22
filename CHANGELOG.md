# Changelog

## 0.1.0

- **Qdrant backend** — replaced JSON file storage with Qdrant vector database for persistence and search
- **Collections** — first-class collection/namespace support via `/rag collection list|create|delete|use`
- **Collection-aware tools** — `rag_index`, `rag_query`, `rag_status` all accept an optional `collection` parameter
- **Collection-aware commands** — `/rag index`, `/rag search`, `/rag status`, `/rag rebuild`, `/rag clear` all accept `--collection <name>`
- **Default collection** — configurable via `/rag collection use <name>`; used for auto-injection and when no collection is specified
- **Qdrant vector search + BM25 re-rank** — oversample from Qdrant then do BM25 scoring on candidates for hybrid results
- **Batched upserts** — points are batched (64 at a time) for efficient indexing
- **Filter-based file deletion** — removes old chunks by file path filter before re-indexing changed files
- **Environment config** — `PI_QDRANT_URL` to point at a custom Qdrant instance (default: `http://127.0.0.1:6333`)

### Inherited from pi-local-rag v0.3.0

- Same chunking, file collection, embeddings (Transformers.js ONNX), BM25 scoring, auto-injection hook
- Same `/rag ext list|add|remove|reset`, `/rag on|off`, progress UI
- Same default file extensions and skip directories
