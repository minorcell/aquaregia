import Link from "next/link";
import { PostCard } from "nextra-theme-blog";
import { notFound } from "next/navigation";
import { getBlogPosts } from "../../../lib/content-index";
import { getLocaleStrings, isValidLocale } from "../../../lib/site-config";

export async function generateMetadata({ params }) {
  const { lang } = await params;
  const t = getLocaleStrings(lang);

  return {
    title: t.blogTitle,
    description:
      lang === "zh"
        ? "Aquaregia 更新、设计与实践文章"
        : "Aquaregia updates, architecture notes, and best practices",
  };
}

export default async function BlogIndexPage({ params }) {
  const { lang } = await params;
  if (!isValidLocale(lang)) {
    notFound();
  }

  const posts = await getBlogPosts(lang);

  return (
    <main>
      <h1>{lang === "zh" ? "博客" : "Blog"}</h1>
      <p>
        {lang === "zh"
          ? "发布日志、架构决策和生产实践。"
          : "Release notes, architecture decisions, and production patterns."}
      </p>

      <p>
        <Link href={`/${lang}/blog/tags`}>
          {lang === "zh" ? "按标签浏览" : "Browse by tags"}
        </Link>
      </p>

      {posts.map((post) => (
        <PostCard
          key={post.route}
          post={post}
          readMore={lang === "zh" ? "继续阅读 ->" : "Read more ->"}
        />
      ))}
    </main>
  );
}
