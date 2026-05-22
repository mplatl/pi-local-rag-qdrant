/**
 * Patch @qdrant/js-client-rest to not use a custom undici Agent dispatcher.
 *
 * When pi replaces the global fetch with its own undici version
 * (via undici.install()), passing a custom dispatcher from a different
 * undici module instance causes "fetch failed / invalid onError method".
 *
 * This patch makes the Qdrant client rely on the global dispatcher instead.
 */
const fs = require("node:fs");
const path = require("node:path");

const variants = ["esm", "cjs"];
const apiClientRel = "api-client.js";
const qdrantDist = path.resolve(
  __dirname,
  "..",
  "node_modules",
  "@qdrant",
  "js-client-rest",
  "dist",
);

for (const variant of variants) {
  const filePath = path.join(qdrantDist, variant, apiClientRel);
  if (!fs.existsSync(filePath)) continue;

  let content = fs.readFileSync(filePath, "utf8");

  // Replace the dispatcher assignment: conditionally creates an Agent,
  // but we set it to undefined so the global dispatcher is used.
  const patched = content.replace(
    /dispatcher:\s*typeof process !== 'undefined'\s*&&[\s\S]*?\n\s*\?[\s\S]*?\n\s*: undefined,/,
    "dispatcher: undefined,",
  );

  if (patched !== content) {
    fs.writeFileSync(filePath, patched);
    console.log(`[pi-local-rag-qdrant] patched ${variant}/${apiClientRel}`);
  } else {
    console.log(
      `[pi-local-rag-qdrant] ${variant}/${apiClientRel} already patched`,
    );
  }
}
