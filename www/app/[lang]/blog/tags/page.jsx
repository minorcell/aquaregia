import Link from "next/link";
import { notFound } from "next/navigation";
import { getBlogTags, slugifyTag } from "../../../../lib/content-index";
import { isValidLocale } from "../../../../lib/site-config";

export const dynamic = "force-static";

export default async function BlogTagsPage({ params }) {
  const { lang } = await params;
  if (!isValidLocale(lang)) {
    notFound();
  }

  const tags = await getBlogTags(lang);

  return (
    <main>
      <h1>{lang === "zh" ? "标签" : "Tags"}</h1>
      <ul>
        {tags.map(({ tag, count }) => (
          <li key={tag}>
            <Link href={`/${lang}/blog/tags/${slugifyTag(tag)}`}>
              {tag} ({count})
            </Link>
          </li>
        ))}
      </ul>
    </main>
  );
}
