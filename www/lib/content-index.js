import { promises as fs } from "node:fs";
import path from "node:path";
import matter from "gray-matter";
import { BlogFrontmatterSchema } from "./contracts";

const CONTENT_ROOT = path.join(process.cwd(), "content");

async function walk(dir) {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  const files = [];

  for (const entry of entries) {
    const entryPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await walk(entryPath)));
      continue;
    }
    files.push(entryPath);
  }

  return files;
}

function toRoute(lang, relativePath) {
  const cleaned = relativePath
    .replace(/\\/g, "/")
    .replace(/\.(md|mdx)$/i, "")
    .replace(/\/index$/, "");

  return `/${lang}/blog/${cleaned}`.replace(/\/+/g, "/");
}

function toSlug(relativePath) {
  return relativePath
    .replace(/\\/g, "/")
    .replace(/\.(md|mdx)$/i, "")
    .replace(/\/index$/, "");
}

function maybeDate(value) {
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? new Date(0) : d;
}

export async function getBlogPosts(lang) {
  const baseDir = path.join(CONTENT_ROOT, lang, "blog", "posts");

  try {
    await fs.access(baseDir);
  } catch {
    return [];
  }

  const files = (await walk(baseDir)).filter(
    (file) => /\.(md|mdx)$/i.test(file) && !path.basename(file).startsWith("_"),
  );

  const posts = await Promise.all(
    files.map(async (file) => {
      const raw = await fs.readFile(file, "utf8");
      const parsed = matter(raw);
      const frontMatter = BlogFrontmatterSchema.parse(parsed.data);
      const relative = path.relative(baseDir, file);

      return {
        slug: toSlug(relative),
        route: toRoute(lang, path.join("posts", relative)),
        frontMatter,
      };
    }),
  );

  return posts.sort(
    (a, b) =>
      maybeDate(b.frontMatter.date).getTime() -
      maybeDate(a.frontMatter.date).getTime(),
  );
}

export async function getBlogTags(lang) {
  const posts = await getBlogPosts(lang);
  const counts = new Map();

  for (const post of posts) {
    for (const tag of post.frontMatter.tags || []) {
      counts.set(tag, (counts.get(tag) || 0) + 1);
    }
  }

  return [...counts.entries()]
    .map(([tag, count]) => ({ tag, count }))
    .sort((a, b) => b.count - a.count || a.tag.localeCompare(b.tag));
}

export async function getPostsByTag(lang, tag) {
  const posts = await getBlogPosts(lang);
  return posts.filter((post) => (post.frontMatter.tags || []).includes(tag));
}

export function slugifyTag(tag) {
  return encodeURIComponent(tag.toLowerCase());
}

export function deslugifyTag(tagSlug) {
  return decodeURIComponent(tagSlug);
}
