import { LOCALES, SITE_URL } from "../lib/site-config";

const coreRoutes = ["", "/docs", "/blog", "/blog/tags", "/search"];

export default function sitemap() {
  const now = new Date();
  return LOCALES.flatMap((lang) =>
    coreRoutes.map((route) => ({
      url: `${SITE_URL}/${lang}${route}`,
      lastModified: now,
      changeFrequency: route.startsWith("/blog") ? "daily" : "weekly",
      priority: route === "" ? 1 : 0.7,
    })),
  );
}
