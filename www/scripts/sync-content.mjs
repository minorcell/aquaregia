import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SyncManifestSchema } from "../lib/contracts.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const wwwRoot = path.resolve(__dirname, "..");
const repoRoot = path.resolve(wwwRoot, "..");

const manifest = SyncManifestSchema.parse([
  {
    source: "README_CN.md",
    targetLanguage: "zh",
    targetPath: "docs/synced/readme-cn.mdx",
    strategy: "replace",
  },
  {
    source: "README.md",
    targetLanguage: "en",
    targetPath: "docs/synced/readme.mdx",
    strategy: "replace",
  },
  {
    source: "examples/README.md",
    targetLanguage: "zh",
    targetPath: "docs/synced/examples.mdx",
    strategy: "replace",
  },
  {
    source: "examples/README.md",
    targetLanguage: "en",
    targetPath: "docs/synced/examples.mdx",
    strategy: "replace",
  },
]);

function deriveTitle(content, fallback) {
  const match = content.match(/^#\s+(.+)$/m);
  return match?.[1]?.trim() || fallback;
}

function deriveDescription(content, fallback) {
  const lines = content
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .filter((line) => !line.startsWith("#"));

  return lines[0]?.slice(0, 140) || fallback;
}

function ensureFrontmatter(content, language, fallbackTitle) {
  if (content.startsWith("---\n")) {
    return content;
  }

  const title = deriveTitle(content, fallbackTitle);
  const description = deriveDescription(
    content,
    language === "zh" ? "自动同步文档" : "Auto-synced docs",
  );

  return [
    "---",
    `title: ${JSON.stringify(title)}`,
    `description: ${JSON.stringify(description)}`,
    `locale: ${language}`,
    "searchable: true",
    "---",
    "",
    content,
  ].join("\n");
}

function normalizeForMdx(content) {
  // MDX doesn't accept HTML comments. Convert them into JSX comments.
  return content.replace(/<!--([\s\S]*?)-->/g, (_m, inner) => `{/*${inner.trim()}*/}`);
}

async function ensureDir(filePath) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
}

async function fileExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function syncManifestEntries() {
  let synced = 0;

  for (const entry of manifest) {
    const sourcePath = path.join(repoRoot, entry.source);
    if (!(await fileExists(sourcePath))) {
      continue;
    }

    const targetPath = path.join(wwwRoot, "content", entry.targetLanguage, entry.targetPath);

    const raw = await fs.readFile(sourcePath, "utf8");
    const hydrated = normalizeForMdx(
      ensureFrontmatter(
      raw,
      entry.targetLanguage,
      path.basename(entry.source, path.extname(entry.source)),
      ),
    );

    await ensureDir(targetPath);
    await fs.writeFile(targetPath, hydrated, "utf8");
    synced += 1;
  }

  return synced;
}

async function copyRepoDocs(lang) {
  const sourceDir = path.join(repoRoot, "docs", lang);
  const targetDir = path.join(wwwRoot, "content", lang, "docs", "synced", "repo-docs");

  if (!(await fileExists(sourceDir))) {
    return 0;
  }

  let copied = 0;

  async function walk(current, relative = "") {
    const entries = await fs.readdir(current, { withFileTypes: true });

    for (const entry of entries) {
      const sourcePath = path.join(current, entry.name);
      const rel = path.join(relative, entry.name);
      const targetPath = path.join(targetDir, rel).replace(/\.(md)$/i, ".mdx");

      if (entry.isDirectory()) {
        await walk(sourcePath, rel);
        continue;
      }

      if (!/\.(md|mdx)$/i.test(entry.name)) {
        continue;
      }

      const content = await fs.readFile(sourcePath, "utf8");
      const hydrated = normalizeForMdx(
        ensureFrontmatter(
          content,
          lang,
          path.basename(entry.name, path.extname(entry.name)),
        ),
      );
      await ensureDir(targetPath);
      await fs.writeFile(targetPath, hydrated, "utf8");
      copied += 1;
    }
  }

  await walk(sourceDir);
  return copied;
}

async function main() {
  const fromManifest = await syncManifestEntries();
  const zhDocs = await copyRepoDocs("zh");
  const enDocs = await copyRepoDocs("en");

  console.log(
    `[sync-content] synced=${fromManifest} repo-docs-zh=${zhDocs} repo-docs-en=${enDocs}`,
  );
}

main().catch((error) => {
  console.error("[sync-content] failed:", error);
  process.exit(1);
});
