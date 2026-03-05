import { getBlogPosts } from "../../../../lib/content-index";
import { isValidLocale, SITE_URL } from "../../../../lib/site-config";

export async function GET(_request, context) {
  const { lang } = await context.params;
  if (!isValidLocale(lang)) {
    return new Response("Not Found", { status: 404 });
  }

  const posts = await getBlogPosts(lang);

  const items = posts
    .map((post) => {
      const link = `${SITE_URL}${post.route}`;
      const date = new Date(post.frontMatter.date).toUTCString();
      return `<item><title><![CDATA[${post.frontMatter.title}]]></title><link>${link}</link><guid>${link}</guid><pubDate>${date}</pubDate><description><![CDATA[${post.frontMatter.description}]]></description></item>`;
    })
    .join("");

  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<title>Aquaregia Blog (${lang})</title>
<link>${SITE_URL}/${lang}/blog</link>
<description>Aquaregia blog feed</description>
${items}
</channel></rss>`;

  return new Response(xml, {
    headers: {
      "Content-Type": "application/rss+xml; charset=utf-8",
      "Cache-Control": "public, s-maxage=3600, stale-while-revalidate=86400",
    },
  });
}
