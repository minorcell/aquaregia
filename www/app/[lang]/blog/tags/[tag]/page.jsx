import Link from "next/link";
import { PostCard } from "nextra-theme-blog";
import { notFound } from "next/navigation";
import {
  deslugifyTag,
  getBlogTags,
  getPostsByTag,
  slugifyTag,
} from "../../../../../lib/content-index";
import { LOCALES, isValidLocale } from "../../../../../lib/site-config";

export async function generateStaticParams() {
  const params = [];

  for (const lang of LOCALES) {
    const tags = await getBlogTags(lang);
    for (const { tag } of tags) {
      params.push({ lang, tag: slugifyTag(tag) });
    }
  }

  return params;
}

export default async function PostsByTagPage({ params }) {
  const { lang, tag: tagSlug } = await params;
  if (!isValidLocale(lang)) {
    notFound();
  }

  const tag = deslugifyTag(tagSlug);
  const posts = await getPostsByTag(lang, tag);

  if (!posts.length) {
    notFound();
  }

  return (
    <main>
      <h1>
        {lang === "zh" ? "标签" : "Tag"}: {tag}
      </h1>
      <p>
        <Link href={`/${lang}/blog/tags`}>
          {lang === "zh" ? "返回标签列表" : "Back to all tags"}
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
