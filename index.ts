/**
 * pi-local-rag-qdrant — Hybrid RAG Pipeline backed by Qdrant
 *
 * Fork of pi-local-rag with Qdrant as the vector backend, adding first-class
 * collection/namespace support so you can maintain separate indexes per project.
 *
 * Index local files → chunk → embed → store in Qdrant → retrieve → inject into LLM context.
 * Uses Transformers.js (ONNX) for local embeddings — zero cloud dependency.
 * Requires a local Qdrant instance (docker run -p 6333:6333 qdrant/qdrant).
 *
 * /rag index <path> [--collection <name>]    → index + embed a file or directory
 * /rag search <query> [--collection <name>]  → hybrid search (BM25 + vector)
 * /rag status [--collection <name>]          → show index stats
 * /rag rebuild [--collection <name>]         → rebuild entire index
 * /rag clear [--collection <name>]           → clear a collection or all
 * /rag on|off                                → toggle auto-injection
 * /rag ext list|add|remove|reset             → configure indexable file extensions
 * /rag collection list|create|delete|use     → manage collections
 *
 * Tools: rag_index, rag_query, rag_status
 */
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { Type } from "@sinclair/typebox";
import { QdrantClient } from "@qdrant/js-client-rest";
import { existsSync, readFileSync, writeFileSync, mkdirSync, readdirSync, statSync } from "node:fs";
import { join, extname, basename } from "node:path";
import { homedir } from "node:os";
import { createHash } from "node:crypto";

// ─── Constants ───────────────────────────────────────────────────────────────

const QDRANT_URL = process.env.PI_QDRANT_URL ?? "http://127.0.0.1:6333";
const RAG_DIR = join(homedir(), ".pi", "rag-qdrant");
const CONFIG_FILE = join(RAG_DIR, "config.json");
const META_FILE = join(RAG_DIR, "collections.json");
const DEFAULT_COLLECTION = "default";

const RST = "\x1b[0m", B = "\x1b[1m", D = "\x1b[2m";
const GREEN = "\x1b[32m", YELLOW = "\x1b[33m", CYAN = "\x1b[36m", RED = "\x1b[31m", MAGENTA = "\x1b[35m";

const EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2";
const VECTOR_DIM = 384;

export const DEFAULT_TEXT_EXTS = [
  ".md", ".mdx", ".txt", ".rst",
  ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
  ".py", ".rs", ".go", ".java", ".kt", ".kts", ".scala",
  ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx",
  ".cs", ".fs", ".vb",
  ".swift", ".m", ".mm",
  ".rb", ".php", ".pl", ".lua", ".dart", ".ex", ".exs", ".erl", ".clj", ".cljs", ".edn",
  ".vue", ".svelte", ".astro",
  ".css", ".scss", ".sass", ".less",
  ".html", ".htm",
  ".json", ".jsonc", ".yaml", ".yml", ".toml", ".ini", ".xml", ".csv", ".tsv",
  ".sh", ".bash", ".zsh", ".fish", ".ps1",
  ".sql", ".graphql", ".gql", ".proto",
  ".env", ".gitignore", ".dockerfile", ".tf", ".hcl",
];

const SKIP_DIRS = new Set([
  "node_modules", ".git", ".next", "dist", "build", "__pycache__", ".venv", "venv", ".cache",
]);

// ─── Types ───────────────────────────────────────────────────────────────────

interface ChunkPayload {
  file: string;
  content: string;
  lineStart: number;
  lineEnd: number;
  hash: string;
  indexed: string;
  tokens: number;
}

interface FileMeta {
  hash: string;
  chunks: number;
  indexed: string;
  size: number;
}

interface CollectionMeta {
  files: Record<string, FileMeta>;
  lastBuild: string;
  embeddingModel?: string;
  vectorCount?: number;
}

interface CollectionsMeta {
  [collectionName: string]: CollectionMeta;
}

interface RagConfig {
  ragEnabled: boolean;
  ragTopK: number;
  ragScoreThreshold: number;
  ragAlpha: number; // 0 = pure vector, 1 = pure BM25
  extraExtensions: string[];
  excludeExtensions: string[];
  defaultCollection: string;
}

interface ScoredChunk {
  id: string;
  file: string;
  content: string;
  lineStart: number;
  lineEnd: number;
  tokens: number;
  vector: number;  // normalized cosine similarity
  bm25: number;    // normalized BM25 score
  hybrid: number;  // combined score
}

// ─── Qdrant Client ───────────────────────────────────────────────────────────

let _qdrant: QdrantClient | null = null;

function getQdrant(): QdrantClient {
  if (!_qdrant) _qdrant = new QdrantClient({ url: QDRANT_URL });
  return _qdrant;
}

// ─── Config ──────────────────────────────────────────────────────────────────

export function loadConfig(): RagConfig {
  ensureDir();
  if (!existsSync(CONFIG_FILE)) return defaultConfig();
  try {
    return { ...defaultConfig(), ...JSON.parse(readFileSync(CONFIG_FILE, "utf-8")) };
  } catch { return defaultConfig(); }
}

function defaultConfig(): RagConfig {
  return {
    ragEnabled: true,
    ragTopK: 5,
    ragScoreThreshold: 0.1,
    ragAlpha: 0.4,
    extraExtensions: [],
    excludeExtensions: [],
    defaultCollection: DEFAULT_COLLECTION,
  };
}

export function normalizeExt(ext: string): string {
  const trimmed = ext.trim().toLowerCase();
  if (!trimmed) return "";
  return trimmed.startsWith(".") ? trimmed : `.${trimmed}`;
}

export function resolveExtensions(config: Pick<RagConfig, "extraExtensions" | "excludeExtensions">): Set<string> {
  const set = new Set(DEFAULT_TEXT_EXTS);
  for (const e of config.extraExtensions) {
    const n = normalizeExt(e);
    if (n) set.add(n);
  }
  for (const e of config.excludeExtensions) {
    const n = normalizeExt(e);
    if (n) set.delete(n);
  }
  return set;
}

export function saveConfig(config: RagConfig) {
  ensureDir();
  writeFileSync(CONFIG_FILE, JSON.stringify(config, null, 2));
}

// ─── Collections Metadata ────────────────────────────────────────────────────

function ensureDir() {
  if (!existsSync(RAG_DIR)) mkdirSync(RAG_DIR, { recursive: true });
}

export function loadMeta(): CollectionsMeta {
  ensureDir();
  if (!existsSync(META_FILE)) return {};
  try { return JSON.parse(readFileSync(META_FILE, "utf-8")); }
  catch { return {}; }
}

export function saveMeta(meta: CollectionsMeta) {
  ensureDir();
  writeFileSync(META_FILE, JSON.stringify(meta, null, 2));
}

// ─── Qdrant Collection Management ────────────────────────────────────────────

async function ensureCollection(name: string): Promise<void> {
  const qdrant = getQdrant();
  try {
    await qdrant.getCollection(name);
  } catch {
    await qdrant.createCollection(name, {
      vectors: { size: VECTOR_DIM, distance: "Cosine" },
    });
    // Create payload index on 'file' field for efficient filter-based deletion during re-indexing
    try {
      await qdrant.createPayloadIndex(name, {
        field_name: "file",
        field_schema: "keyword",
        wait: true,
      });
    } catch {
      // Index may already exist or not be supported — safe to ignore
    }
  }
}

async function listCollections(): Promise<string[]> {
  const qdrant = getQdrant();
  const result = await qdrant.getCollections();
  return result.collections.map((c: { name: string }) => c.name);
}

async function deleteCollection(name: string): Promise<void> {
  const qdrant = getQdrant();
  await qdrant.deleteCollection(name);
}

async function getCollectionInfo(name: string): Promise<{
  pointsCount: number;
  vectorsCount: number;
  segmentsCount: number;
}> {
  const qdrant = getQdrant();
  const result = await qdrant.getCollection(name);
  return {
    pointsCount: result.points_count ?? 0,
    vectorsCount: result.indexed_vectors_count ?? 0,
    segmentsCount: result.segments_count ?? 0,
  };
}

export async function collectionExists(name: string): Promise<boolean> {
  try {
    await getQdrant().getCollection(name);
    return true;
  } catch { return false; }
}

// ─── Embeddings ──────────────────────────────────────────────────────────────

let _pipeline: any = null;

async function getEmbedder() {
  if (_pipeline) return _pipeline;
  const { pipeline } = await import("@xenova/transformers");
  _pipeline = await pipeline("feature-extraction", EMBEDDING_MODEL);
  return _pipeline;
}

export async function embed(text: string): Promise<number[]> {
  const embedder = await getEmbedder();
  const output = await embedder(text, { pooling: "mean", normalize: true });
  return Array.from(output.data as Float32Array);
}

async function embedBatch(texts: string[], onProgress?: (i: number, total: number) => void): Promise<number[][]> {
  const results: number[][] = [];
  for (let i = 0; i < texts.length; i++) {
    results.push(await embed(texts[i]));
    onProgress?.(i + 1, texts.length);
  }
  return results;
}

// ─── Math ────────────────────────────────────────────────────────────────────

export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

export function normalize(scores: number[]): number[] {
  const max = Math.max(...scores);
  const min = Math.min(...scores);
  const range = max - min;
  if (range === 0) return scores.map(s => s > 0 ? 1 : 0);
  return scores.map(s => (s - min) / range);
}

// ─── Chunking & File Collection ──────────────────────────────────────────────

export function chunkText(text: string, maxLines = 50): { content: string; lineStart: number; lineEnd: number }[] {
  const lines = text.split("\n");
  const chunks: { content: string; lineStart: number; lineEnd: number }[] = [];
  let i = 0;
  while (i < lines.length) {
    let end = Math.min(i + maxLines, lines.length);
    // Try to break at a blank line near the end
    for (let j = end - 1; j > i + 10 && j > end - 15; j--) {
      if (lines[j]?.trim() === "") { end = j + 1; break; }
    }
    const chunk = lines.slice(i, end).join("\n");
    if (chunk.trim().length > 20) {
      chunks.push({ content: chunk, lineStart: i + 1, lineEnd: end });
    }
    i = end;
  }
  return chunks;
}

function pointId(filePath: string, lineStart: number): string {
  const h = createHash("sha256").update(`${filePath}:${lineStart}`).digest("hex");
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-4${h.slice(13, 16)}-${(parseInt(h[16], 16) | 0x8).toString(16)}${h.slice(17, 20)}-${h.slice(20, 32)}`;
}

function sha256(data: string): string {
  return createHash("sha256").update(data).digest("hex").slice(0, 12);
}

export function collectFiles(dirPath: string, exts?: Set<string>): string[] {
  const allowed = exts ?? resolveExtensions(loadConfig());
  const files: string[] = [];
  function walk(dir: string) {
    try {
      for (const entry of readdirSync(dir, { withFileTypes: true })) {
        if (entry.isDirectory()) {
          if (!SKIP_DIRS.has(entry.name) && !entry.name.startsWith(".")) walk(join(dir, entry.name));
        } else if (allowed.has(extname(entry.name).toLowerCase())) {
          const fp = join(dir, entry.name);
          try {
            if (statSync(fp).size < 500_000) files.push(fp);
          } catch {}
        }
      }
    } catch {}
  }
  try {
    const st = statSync(dirPath);
    if (st.isFile()) {
      if (!allowed.has(extname(dirPath).toLowerCase()) || st.size >= 500_000) return [];
      return [dirPath];
    }
  } catch { return []; }
  walk(dirPath);
  return files;
}

// ─── Indexing ─────────────────────────────────────────────────────────────────

interface ProgressCallbacks {
  onFile?: (current: number, total: number, filename: string, skipped: number) => void;
  onChunk?: (fileChunk: number, totalChunks: number, filename: string) => void;
  onSave?: () => void;
}

const yield_ = () => new Promise<void>(r => setTimeout(r, 0));

function stderrProgress(msg: string) {
  process.stderr.write(`\r\x1b[2K${msg}`);
}

async function indexFiles(
  paths: string[],
  collection: string,
  progress?: ProgressCallbacks
): Promise<{ indexed: number; chunks: number; skipped: number; durationMs: number }> {
  await ensureCollection(collection);

  const meta = loadMeta();
  if (!meta[collection]) meta[collection] = { files: {}, lastBuild: "" };
  const collMeta = meta[collection];

  const qdrant = getQdrant();
  let indexed = 0, totalChunks = 0, skipped = 0;
  const startMs = Date.now();
  const total = paths.length;

  // Collect all points to upsert in batches for efficiency
  let pointBatch: Array<{ id: string; vector: number[]; payload: Record<string, unknown> }> = [];

  const BATCH_SIZE = 64;

  async function flushBatch() {
    if (pointBatch.length === 0) return;
    await qdrant.upsert(collection, { wait: true, points: pointBatch });
    pointBatch = [];
  }

  for (let i = 0; i < paths.length; i++) {
    const fp = paths[i];
    const pct = Math.round(((i + 1) / total) * 100);
    const name = basename(fp);

    try {
      const content = readFileSync(fp, "utf-8");
      const hash = sha256(content);

      // Skip unchanged files
      if (collMeta.files[fp]?.hash === hash) {
        skipped++;
        stderrProgress(`[${i + 1}/${total}] ${pct}% skipped ${name}`);
        progress?.onFile?.(i + 1, total, name, skipped);
        await yield_();
        continue;
      }

      // Delete old points for this file (Qdrant filter-based deletion)
      try {
        await qdrant.delete(collection, {
          filter: { must: [{ key: "file", match: { value: fp } }] },
        });
      } catch {
        // Collection may be empty or filter may have no matches — ignore
      }

      const rawChunks = chunkText(content);

      stderrProgress(`[${i + 1}/${total}] ${pct}% embedding ${name} (${rawChunks.length} chunks)`);
      progress?.onFile?.(i + 1, total, name, skipped);
      await yield_();

      const vectors = await embedBatch(
        rawChunks.map(c => c.content),
        (ci) => {
          stderrProgress(`[${i + 1}/${total}] ${pct}% ${name} — chunk ${ci}/${rawChunks.length}`);
          progress?.onChunk?.(ci, rawChunks.length, name);
        }
      );

      for (let j = 0; j < rawChunks.length; j++) {
        const chunk = rawChunks[j];
        pointBatch.push({
          id: pointId(fp, chunk.lineStart),
          vector: vectors[j],
          payload: {
            file: fp,
            content: chunk.content,
            lineStart: chunk.lineStart,
            lineEnd: chunk.lineEnd,
            hash: sha256(chunk.content),
            indexed: new Date().toISOString(),
            tokens: Math.ceil(chunk.content.length / 4),
          },
        });

        if (pointBatch.length >= BATCH_SIZE) await flushBatch();
      }

      collMeta.files[fp] = {
        hash,
        chunks: rawChunks.length,
        indexed: new Date().toISOString(),
        size: content.length,
      };
      indexed++;
      totalChunks += rawChunks.length;
    } catch { skipped++; }
  }

  // Flush remaining points
  await flushBatch();

  stderrProgress("");
  process.stderr.write(`\r\x1b[2K`);

  progress?.onSave?.();

  collMeta.lastBuild = new Date().toISOString();
  collMeta.embeddingModel = EMBEDDING_MODEL;

  // Update vector count from Qdrant
  try {
    const info = await getCollectionInfo(collection);
    collMeta.vectorCount = info.vectorsCount;
  } catch {
    collMeta.vectorCount = totalChunks;
  }

  saveMeta(meta);
  return { indexed, chunks: totalChunks, skipped, durationMs: Date.now() - startMs };
}

// ─── Search ───────────────────────────────────────────────────────────────────

export async function hybridSearch(
  query: string,
  collection: string,
  limit = 10,
  alpha = 0.4,
  threshold = 0.1
): Promise<ScoredChunk[]> {
  const qdrant = getQdrant();

  // Check if collection exists
  if (!(await collectionExists(collection))) return [];

  const queryVec = await embed(query);

  // Oversample from Qdrant to give BM25 more candidates to re-rank
  let qdrantResults: Array<{
    id: string | number;
    score: number;
    payload?: Record<string, unknown> | null;
  }>;

  try {
    qdrantResults = await qdrant.search(collection, {
      vector: queryVec,
      limit: Math.max(limit * 5, 50),
      with_payload: true,
      with_vector: false,
      score_threshold: 0, // get all, we do our own thresholding
    });
  } catch {
    return [];
  }

  if (!qdrantResults.length) return [];

  // Extract chunks from Qdrant payloads
  const chunks: ScoredChunk[] = qdrantResults
    .filter(r => r.payload)
    .map(r => ({
      id: String(r.id),
      file: String(r.payload!.file ?? ""),
      content: String(r.payload!.content ?? ""),
      lineStart: Number(r.payload!.lineStart ?? 1),
      lineEnd: Number(r.payload!.lineEnd ?? 1),
      tokens: Number(r.payload!.tokens ?? 0),
      vector: r.score, // Qdrant cosine similarity score
      bm25: 0,
      hybrid: 0,
    }));

  // ── BM25 re-ranking on Qdrant results ──
  const terms = query.toLowerCase().split(/\s+/).filter(t => t.length > 1);
  const queryLower = query.toLowerCase();

  const idfMap = new Map<string, number>();
  for (const term of terms) {
    const docsWithTerm = chunks.filter(c => c.content.toLowerCase().includes(term)).length;
    idfMap.set(term, Math.log(1 + chunks.length / (1 + docsWithTerm)));
  }

  const bm25Raw = chunks.map(chunk => {
    const lower = chunk.content.toLowerCase();
    let score = 0;
    for (const term of terms) {
      const count = (lower.match(new RegExp(term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "g")) || []).length;
      if (count > 0) score += Math.log(1 + count) * (idfMap.get(term) ?? 0);
    }
    if (lower.includes(queryLower)) score *= 2;       // exact phrase boost
    if (chunk.file.toLowerCase().includes(terms[0] ?? "")) score *= 1.5; // filename boost
    return score;
  });

  const bm25Norm = normalize(bm25Raw);
  const vectorNorm = normalize(chunks.map(c => c.vector));

  // ── Hybrid scoring ──
  for (let i = 0; i < chunks.length; i++) {
    chunks[i].bm25 = bm25Norm[i];
    chunks[i].vector = vectorNorm[i];
    chunks[i].hybrid = alpha * bm25Norm[i] + (1 - alpha) * vectorNorm[i];
  }

  return chunks
    .filter(s => s.hybrid >= threshold)
    .sort((a, b) => b.hybrid - a.hybrid)
    .slice(0, limit);
}

// ─── Extension ────────────────────────────────────────────────────────────────

export default function (pi: ExtensionAPI) {
  ensureDir();

  // ── Auto-inject RAG context before every agent turn ──
  pi.on("before_agent_start", async (event, _ctx) => {
    const config = loadConfig();
    if (!config.ragEnabled) return;

    const collection = config.defaultCollection;
    if (!(await collectionExists(collection))) return;

    const results = await hybridSearch(
      event.prompt,
      collection,
      config.ragTopK,
      config.ragAlpha,
      config.ragScoreThreshold
    );
    if (!results.length) return;

    const context = results.map(r =>
      `### ${basename(r.file)} (lines ${r.lineStart}-${r.lineEnd})  score=${r.hybrid.toFixed(2)}\n` +
      `\`\`\`\n${r.content.slice(0, 600)}\n\`\`\``
    ).join("\n\n");

    return {
      systemPrompt: event.systemPrompt +
        `\n\n## Relevant Codebase Context (pi-local-rag-qdrant · collection: ${collection})\n` +
        `*Retrieved ${results.length} chunks via hybrid search (BM25 + vector)*\n\n` +
        context,
    };
  });

  // ── /rag command ──
  pi.registerCommand("rag", {
    description: "pi-local-rag-qdrant: /rag index|search|status|rebuild|clear|on|off|ext|collection",
    handler: async (args, ctx) => {
      const parts = (args || "").trim().split(/\s+/);
      const cmd = parts[0] || "status";

      // Helper: extract --collection flag from args
      function extractCollection(parts: string[]): { collection: string; rest: string[] } {
        const collIdx = parts.indexOf("--collection");
        if (collIdx >= 0 && collIdx + 1 < parts.length) {
          const col = parts[collIdx + 1];
          return { collection: col, rest: parts.filter((_, i) => i !== collIdx && i !== collIdx + 1) };
        }
        return { collection: loadConfig().defaultCollection, rest: parts };
      }

      // ── collection management ──
      if (cmd === "collection" || cmd === "col") {
        const sub = (parts[1] || "list").toLowerCase();
        const th = ctx.ui.theme;

        if (sub === "list") {
          const collections = await listCollections();
          const meta = loadMeta();
          const config = loadConfig();
          const lines: string[] = [
            th.bold("📂 Collections"),
            "",
          ];
          if (collections.length === 0) {
            lines.push(th.fg("dim", "  (no collections)"));
          }
          for (const name of collections.sort()) {
            const isDefault = name === config.defaultCollection;
            const info = meta[name];
            const fileCount = info ? Object.keys(info.files).length : 0;
            const suffix = isDefault ? "  " + th.fg("success", "◀ default") : "";
            lines.push(`  ${th.fg("accent", name)}  ${th.fg("dim", `${fileCount} files`)}${suffix}`);
          }
          lines.push("");
          lines.push(th.fg("dim", "Edit via /rag collection create|delete|use <name>"));
          ctx.ui.setWidget("rag-collections", lines);
          return;
        }

        if (sub === "create" || sub === "add") {
          const name = parts[2];
          if (!name) { ctx.ui.notify("Usage: /rag collection create <name>", "warning"); return; }
          if (await collectionExists(name)) {
            ctx.ui.notify(`Collection "${name}" already exists.`, "warning");
            return;
          }
          await ensureCollection(name);
          const meta = loadMeta();
          if (!meta[name]) meta[name] = { files: {}, lastBuild: "" };
          saveMeta(meta);
          ctx.ui.notify(`Collection "${name}" created. Use /rag index --collection ${name} <path> to populate it.`, "info");
          return;
        }

        if (sub === "delete" || sub === "rm") {
          const name = parts[2];
          if (!name) { ctx.ui.notify("Usage: /rag collection delete <name>", "warning"); return; }
          if (!(await collectionExists(name))) {
            ctx.ui.notify(`Collection "${name}" does not exist.`, "warning");
            return;
          }
          const config = loadConfig();
          if (config.defaultCollection === name) {
            ctx.ui.notify(`Cannot delete "${name}" — it's the default collection. Use /rag collection use <other> first.`, "error");
            return;
          }
          await deleteCollection(name);
          const meta = loadMeta();
          delete meta[name];
          saveMeta(meta);
          ctx.ui.notify(`Collection "${name}" deleted.`, "info");
          return;
        }

        if (sub === "use" || sub === "default") {
          const name = parts[2];
          if (!name) { ctx.ui.notify("Usage: /rag collection use <name>", "warning"); return; }
          if (!(await collectionExists(name))) {
            ctx.ui.notify(`Collection "${name}" does not exist. Create it first with /rag collection create ${name}`, "warning");
            return;
          }
          const config = loadConfig();
          config.defaultCollection = name;
          saveConfig(config);
          ctx.ui.notify(`Default collection set to "${name}".`, "info");
          return;
        }

        ctx.ui.notify("Usage: /rag collection list|create <name>|delete <name>|use <name>", "warning");
        return;
      }

      // For index/search/status/rebuild/clear, extract --collection
      const { collection, rest } = extractCollection(parts);

      // ── index ──
      if (cmd === "index") {
        const path = rest[1] || ".";
        if (!existsSync(path)) { ctx.ui.notify(`Path not found: ${path}`, "error"); return; }
        const files = collectFiles(path);
        if (!files.length) { ctx.ui.notify(`No indexable files found in: ${path}`, "warning"); return; }

        const total = files.length;
        ctx.ui.notify(`Found ${total} files to index → collection "${collection}"`, "info");

        function progressBar(n: number, total: number, width = 24): string {
          const filled = Math.round((n / total) * width);
          return CYAN + "█".repeat(filled) + D + "░".repeat(width - filled) + RST;
        }

        const result = await indexFiles(files, collection, {
          onFile(current, total, filename, skipped) {
            const pct = Math.round((current / total) * 100);
            const bar = progressBar(current, total);
            ctx.ui.setStatus("rag", `■ Indexing "${collection}" ${pct}% │ ${current}/${total} files │ ${skipped} unchanged`);
            ctx.ui.setWidget("rag", [
              `${B}${CYAN}Indexing → "${collection}"${RST}  ${bar}  ${GREEN}${pct}%${RST}`,
              `${D}file:    ${RST}${filename}`,
              `${D}done:    ${RST}${GREEN}${current - skipped} embedded${RST}  ${D}${skipped} unchanged${RST}`,
            ]);
          },
          onChunk(ci, total, filename) {
            ctx.ui.setStatus("rag", `■ Embedding ${filename} — chunk ${ci}/${total}`);
          },
          onSave() {
            ctx.ui.setStatus("rag", `■ Saving to Qdrant...`);
          },
        });

        ctx.ui.setStatus("rag", undefined);
        ctx.ui.setWidget("rag", undefined);

        const secs = (result.durationMs / 1000).toFixed(1);
        ctx.ui.notify(`✅ Indexed ${result.indexed} files (${result.chunks} chunks) → "${collection}" · ${result.skipped} unchanged · ${secs}s`, "info");
        return;
      }

      // ── search ──
      if (cmd === "search") {
        const query = rest.slice(1).join(" ");
        if (!query) { ctx.ui.notify("Usage: /rag search <query> [--collection <name>]", "warning"); return; }
        if (!(await collectionExists(collection))) {
          ctx.ui.notify(`Collection "${collection}" does not exist. Run /rag index --collection ${collection} <path> first.`, "warning");
          return;
        }

        const config = loadConfig();
        const results = await hybridSearch(query, collection, 10, config.ragAlpha, config.ragScoreThreshold);
        if (!results.length) { ctx.ui.notify(`No results for "${query}" in "${collection}"`, "warning"); return; }

        const th = ctx.ui.theme;
        const lines: string[] = [
          th.bold(th.fg("accent", "🔍 ") + `${results.length} results for "${query}" in "${collection}"`) +
            "  " + th.fg("dim", "hybrid BM25+vector"),
          "",
        ];
        for (const r of results) {
          lines.push(
            th.fg("success", basename(r.file)) +
            th.fg("muted", `:${r.lineStart}-${r.lineEnd}`) +
            "  " + th.fg("dim", `bm25=${r.bm25.toFixed(2)} vec=${r.vector.toFixed(2)} hybrid=${r.hybrid.toFixed(2)}`)
          );
          const preview = r.content.split("\n").slice(0, 3).join("\n");
          lines.push(th.fg("dim", preview.slice(0, 200)));
          lines.push("");
        }
        ctx.ui.setWidget("rag-search", lines);
        return;
      }

      // ── on/off toggle ──
      if (cmd === "on" || cmd === "off") {
        const config = loadConfig();
        config.ragEnabled = cmd === "on";
        saveConfig(config);
        ctx.ui.notify(cmd === "on" ? "RAG auto-injection enabled" : "RAG auto-injection disabled", "info");
        return;
      }

      // ── rebuild ──
      if (cmd === "rebuild") {
        if (!(await collectionExists(collection))) {
          ctx.ui.notify(`Collection "${collection}" does not exist.`, "warning");
          return;
        }

        const meta = loadMeta();
        const collMeta = meta[collection];
        if (!collMeta || !Object.keys(collMeta.files).length) {
          ctx.ui.notify(`Collection "${collection}" is empty. Run /rag index first.`, "warning");
          return;
        }

        const allFiles = Object.keys(collMeta.files);
        const existingFiles = allFiles.filter(f => existsSync(f));
        const deletedFiles = allFiles.filter(f => !existsSync(f));

        // Prune deleted files from Qdrant and metadata
        for (const f of deletedFiles) {
          delete collMeta.files[f];
        }
        // Force re-embed all existing files
        for (const f of existingFiles) {
          if (collMeta.files[f]) {
            collMeta.files[f] = { ...collMeta.files[f], hash: "" }; // force re-embed
          }
        }
        saveMeta(meta);

        if (deletedFiles.length) ctx.ui.notify(`Pruned ${deletedFiles.length} deleted files from "${collection}"`, "info");
        ctx.ui.notify(`Rebuilding ${existingFiles.length} files in "${collection}"...`, "info");

        function progressBar(n: number, total: number, width = 24): string {
          const filled = Math.round((n / total) * width);
          return CYAN + "█".repeat(filled) + D + "░".repeat(width - filled) + RST;
        }

        const result = await indexFiles(existingFiles, collection, {
          onFile(current, total, filename, skipped) {
            const pct = Math.round((current / total) * 100);
            const bar = progressBar(current, total);
            ctx.ui.setStatus("rag", `■ Rebuilding "${collection}" ${pct}% │ ${current}/${total} │ ${skipped} unchanged`);
            ctx.ui.setWidget("rag", [
              `${B}${CYAN}Rebuilding → "${collection}"${RST}  ${bar}  ${GREEN}${pct}%${RST}`,
              `${D}file:    ${RST}${filename}`,
              `${D}done:    ${RST}${GREEN}${current - skipped} re-embedded${RST}  ${D}${skipped} unchanged${RST}`,
            ]);
          },
          onChunk(ci, total, filename) {
            ctx.ui.setStatus("rag", `■ Embedding ${filename} — chunk ${ci}/${total}`);
          },
          onSave() {
            ctx.ui.setStatus("rag", `■ Saving to Qdrant...`);
          },
        });

        ctx.ui.setStatus("rag", undefined);
        ctx.ui.setWidget("rag", undefined);

        const secs = (result.durationMs / 1000).toFixed(1);
        ctx.ui.notify(`✅ Rebuilt "${collection}": ${result.indexed} re-indexed · ${result.skipped} unchanged · ${deletedFiles.length} deleted · ${result.chunks} chunks · ${secs}s`, "info");
        return;
      }

      // ── ext (configure file extensions) ──
      if (cmd === "ext") {
        const sub = (rest[1] || "list").toLowerCase();
        const config = loadConfig();

        if (sub === "list") {
          const th = ctx.ui.theme;
          const active = Array.from(resolveExtensions(config)).sort();
          const lines: string[] = [
            th.bold("Active file extensions") + "  " + th.fg("dim", `(${active.length})`),
            th.fg("muted", "  " + active.join(" ")),
          ];
          if (config.extraExtensions.length)
            lines.push("  " + th.fg("dim", "extra:   ") + th.fg("success", config.extraExtensions.join(" ")));
          if (config.excludeExtensions.length)
            lines.push("  " + th.fg("dim", "excluded:") + " " + th.fg("warning", config.excludeExtensions.join(" ")));
          lines.push("", th.fg("dim", "Edit via /rag ext add <.ext> / remove <.ext> / reset"));
          ctx.ui.setWidget("rag-ext", lines);
          return;
        }

        if (sub === "add") {
          const ext = normalizeExt(rest[2] || "");
          if (!ext) { ctx.ui.notify("Usage: /rag ext add <.ext>", "warning"); return; }
          config.excludeExtensions = config.excludeExtensions.filter(e => normalizeExt(e) !== ext);
          if (!config.extraExtensions.map(normalizeExt).includes(ext)) config.extraExtensions.push(ext);
          saveConfig(config);
          ctx.ui.notify(`Added ${ext} to indexable extensions. Run /rag index <path> to pick up matching files.`, "info");
          return;
        }

        if (sub === "remove" || sub === "rm") {
          const ext = normalizeExt(rest[2] || "");
          if (!ext) { ctx.ui.notify("Usage: /rag ext remove <.ext>", "warning"); return; }
          const wasExtra = config.extraExtensions.map(normalizeExt).includes(ext);
          config.extraExtensions = config.extraExtensions.filter(e => normalizeExt(e) !== ext);
          if (!wasExtra && !config.excludeExtensions.map(normalizeExt).includes(ext)) config.excludeExtensions.push(ext);
          saveConfig(config);
          ctx.ui.notify(`Removed ${ext} from indexable extensions.`, "info");
          return;
        }

        if (sub === "reset") {
          config.extraExtensions = [];
          config.excludeExtensions = [];
          saveConfig(config);
          ctx.ui.notify("Extension list reset to defaults.", "info");
          return;
        }

        ctx.ui.notify("Usage: /rag ext list|add <.ext>|remove <.ext>|reset", "warning");
        return;
      }

      // ── clear ──
      if (cmd === "clear") {
        if (collection) {
          // Clear a specific collection
          if (await collectionExists(collection)) {
            await deleteCollection(collection);
            await ensureCollection(collection); // recreate empty
          }
          const meta = loadMeta();
          if (meta[collection]) {
            meta[collection] = { files: {}, lastBuild: new Date().toISOString() };
          }
          saveMeta(meta);
          ctx.ui.notify(`Collection "${collection}" cleared.`, "info");
        } else {
          // Clear all
          const all = await listCollections();
          for (const name of all) {
            try { await deleteCollection(name); } catch {}
          }
          saveMeta({});
          ctx.ui.notify("All collections cleared.", "info");
        }
        return;
      }

      // ── status (default) ──
      const th = ctx.ui.theme;
      const config = loadConfig();
      const label = (k: string) => th.fg("dim", k.padEnd(18));
      const val = (v: string | number) => th.fg("success", String(v));

      // If a specific collection is requested, show its details
      const targetCollection = rest[0] === "status" && rest.length > 1 && rest[1] !== "--collection"
        ? rest[1]
        : collection;

      if (targetCollection && rest[0] === "status" && rest.length > 1 && rest[1] !== "--collection") {
        // /rag status <collection> (shorthand without --collection)
        const col = rest[1];
        if (!(await collectionExists(col))) {
          ctx.ui.notify(`Collection "${col}" does not exist.`, "warning");
          return;
        }
        const info = await getCollectionInfo(col);
        const meta = loadMeta();
        const collMeta = meta[col];
        const fileCount = collMeta ? Object.keys(collMeta.files).length : 0;

        const lines: string[] = [
          th.bold(`📂 Collection: ${col}`),
          "",
          "  " + label("Points:") + val(info.pointsCount),
          "  " + label("Vectors:") + val(info.vectorsCount),
          "  " + label("Files tracked:") + val(fileCount),
          "  " + label("Segments:") + val(info.segmentsCount),
          "  " + label("Last build:") + (collMeta?.lastBuild ? val(collMeta.lastBuild) : th.fg("dim", "never")),
          "",
        ];
        ctx.ui.setWidget("rag-status", lines);
        return;
      }

      // Global status: show all collections
      const collections = await listCollections();
      const meta = loadMeta();
      const myCollections = collections.filter(c => meta[c] && Object.keys(meta[c].files).length > 0);
      const totalFiles = myCollections.reduce((s, c) => s + Object.keys(meta[c]?.files ?? {}).length, 0);
      const totalChunks = myCollections.reduce((s, c) => s + (meta[c]?.vectorCount ?? 0), 0);
      const totalTokens = 0; // approximate from chunks

      const lines: string[] = [
        th.bold("🔍 pi-local-rag-qdrant"),
        "",
        "  " + label("Qdrant URL:") + th.fg("dim", QDRANT_URL),
        "  " + label("Collections:") + val(collections.length),
        "  " + th.fg("dim", `  (${myCollections.length} with data)`),
        "  " + label("Files indexed:") + val(totalFiles),
        "  " + label("Total vectors:") + val(totalChunks.toLocaleString()),
        "  " + label("Embedding model:") + th.fg("dim", EMBEDDING_MODEL),
        "  " + label("Storage:") + th.fg("dim", RAG_DIR + " (+ Qdrant)"),
        "",
        "  " + label("RAG injection:") +
          (config.ragEnabled ? th.fg("success", "enabled") : th.fg("warning", "disabled")) +
          th.fg("dim", `  topK=${config.ragTopK}  threshold=${config.ragScoreThreshold}  alpha=${config.ragAlpha}`),
        "  " + label("Default collection:") + th.fg("accent", config.defaultCollection),
      ];

      if (myCollections.length) {
        lines.push("", "  " + th.bold("Collections:"));
        for (const name of myCollections.sort()) {
          const cm = meta[name];
          if (!cm) continue;
          const fc = Object.keys(cm.files).length;
          const isDefault = name === config.defaultCollection;
          const arrow = isDefault ? " " + th.fg("success", "◀") : "";
          lines.push(`    ${th.fg("accent", name)}  ${th.fg("dim", `${fc} files · ${cm.vectorCount ?? "?"} vectors`)}${arrow}`);
        }
      }

      ctx.ui.setWidget("rag-status", lines);
    },
  });

  // ── Tools ──

  pi.registerTool({
    name: "rag_index",
    label: "Index RAG",
    description:"Index a file or directory into a Qdrant-backed RAG collection. Chunks text files, generates embeddings, stores in Qdrant for hybrid BM25+vector search. Use 'collection' to organize separate indexes (e.g. per project).",
    parameters: Type.Object({
      path: Type.String({ description: "File or directory path to index" }),
      collection: Type.Optional(Type.String({ description: "Collection name (default: 'default'). Use separate collections for different projects." })),
    }),
    execute: async (_toolCallId, params) => {
      if (!existsSync(params.path)) return { content: [{ type: "text" as const, text: `Path not found: ${params.path}` }], details: {} };
      const coll = params.collection ?? loadConfig().defaultCollection;
      const files = collectFiles(params.path);
      if (!files.length) return { content: [{ type: "text" as const, text: `No indexable text files found in: ${params.path}` }], details: {} };
      const result = await indexFiles(files, coll, {});
      process.stderr.write(`\n`);
      return { content: [{ type: "text" as const, text: `Indexed ${result.indexed} files (${result.chunks} chunks) → collection "${coll}". ${result.skipped} unchanged. ${(result.durationMs / 1000).toFixed(1)}s` }], details: {} };
    },
  });

  pi.registerTool({
    name: "rag_query",
    label: "Query RAG",
    description:"Search a Qdrant-backed RAG collection using hybrid BM25+vector search. Returns relevant chunks with file paths, line numbers, and relevance scores.",
    parameters: Type.Object({
      query: Type.String({ description: "Search query" }),
      collection: Type.Optional(Type.String({ description: "Collection to search (default: configured default collection)" })),
      limit: Type.Optional(Type.Number({ description: "Max results (default 10)" })),
    }),
    execute: async (_toolCallId, params) => {
      const coll = params.collection ?? loadConfig().defaultCollection;
      if (!(await collectionExists(coll))) return { content: [{ type: "text" as const, text: `Collection "${coll}" does not exist. Run rag_index with a collection parameter first.` }], details: {} };
      const config = loadConfig();
      const results = await hybridSearch(params.query, coll, params.limit ?? 10, config.ragAlpha, config.ragScoreThreshold);
      if (!results.length) return { content: [{ type: "text" as const, text: `No results for "${params.query}" in collection "${coll}"` }], details: {} };
      const text = JSON.stringify(results.map(r => ({
        file: r.file,
        lines: `${r.lineStart}-${r.lineEnd}`,
        tokens: r.tokens,
        scores: { bm25: r.bm25.toFixed(3), vector: r.vector.toFixed(3), hybrid: r.hybrid.toFixed(3) },
        preview: r.content.slice(0, 300),
      })), null, 2);
      return { content: [{ type: "text" as const, text }], details: {} };
    },
  });

  pi.registerTool({
    name: "rag_status",
    label: "RAG Status",
    description:"Show pi-local-rag-qdrant status: Qdrant collections, file counts, vector coverage, RAG config.",
    parameters: Type.Object({
      collection: Type.Optional(Type.String({ description: "Specific collection to inspect" })),
    }),
    execute: async (_toolCallId, params) => {
      const config = loadConfig();
      const collections = await listCollections();
      const meta = loadMeta();

      if (params.collection) {
        const coll = params.collection;
        if (!(await collectionExists(coll))) return { content: [{ type: "text" as const, text: `Collection "${coll}" does not exist.` }], details: {} };
        const info = await getCollectionInfo(coll);
        const cm = meta[coll] ?? { files: {}, lastBuild: "" };
        const text = JSON.stringify({
          collection: coll,
          points: info.pointsCount,
          vectors: info.vectorsCount,
          files: Object.keys(cm.files).length,
          lastBuild: cm.lastBuild || "never",
          embeddingModel: cm.embeddingModel ?? "none",
          isDefault: coll === config.defaultCollection,
        }, null, 2);
        return { content: [{ type: "text" as const, text }], details: {} };
      }

      const activeCollections = collections.filter(c => meta[c] && Object.keys(meta[c].files ?? {}).length > 0);
      const text = JSON.stringify({
        qdrantUrl: QDRANT_URL,
        totalCollections: collections.length,
        activeCollections: activeCollections.length,
        collections: activeCollections.map(c => ({
          name: c,
          files: Object.keys(meta[c].files).length,
          vectors: meta[c].vectorCount ?? "?",
          lastBuild: meta[c].lastBuild || "never",
          isDefault: c === config.defaultCollection,
        })),
        ragConfig: {
          enabled: config.ragEnabled,
          topK: config.ragTopK,
          threshold: config.ragScoreThreshold,
          alpha: config.ragAlpha,
          defaultCollection: config.defaultCollection,
        },
        embeddingModel: EMBEDDING_MODEL,
      }, null, 2);
      return { content: [{ type: "text" as const, text }], details: {} };
    },
  });
}
