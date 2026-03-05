import Link from "next/link";
import { Layout as BlogLayout, Footer, Navbar, ThemeSwitch } from "nextra-theme-blog";
import { getPageMap } from "nextra/page-map";
import { notFound } from "next/navigation";
import {
  LOCALE_LABELS,
  LOCALES,
  SITE_NAME,
  isValidLocale,
} from "../../../lib/site-config";

export default async function BlogRouteLayout({ children, params }) {
  const { lang } = await params;
  if (!isValidLocale(lang)) {
    notFound();
  }

  const pageMap = await getPageMap(`/${lang}/blog`);

  return (
    <BlogLayout>
      <Navbar pageMap={pageMap}>
        <ThemeSwitch />
      </Navbar>
      <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem" }}>
        <p style={{ marginTop: 0 }}>{SITE_NAME} Blog</p>
        <div style={{ display: "flex", gap: ".8rem", marginBottom: "1rem" }}>
          <Link href={`/${lang}/docs`}>Docs</Link>
          <Link href={`/${lang}/search`}>Search</Link>
          {LOCALES.filter((locale) => locale !== lang).map((locale) => (
            <Link key={locale} href={`/${locale}/blog`}>
              {LOCALE_LABELS[locale]}
            </Link>
          ))}
        </div>
      </div>
      {children}
      <Footer>
        MIT {new Date().getFullYear()} © Aquaregia · <Link href={`/${lang}/blog/rss.xml`}>RSS</Link>
      </Footer>
    </BlogLayout>
  );
}
