import Link from "next/link";
import { notFound } from "next/navigation";
import {
  SITE_NAME,
  getLocaleAlternates,
  getLocaleStrings,
  isValidLocale,
} from "../../lib/site-config";

export async function generateMetadata({ params }) {
  const { lang } = await params;
  if (!isValidLocale(lang)) {
    return {};
  }
  const t = getLocaleStrings(lang);

  return {
    title: t.homeTitle,
    alternates: {
      canonical: `/${lang}`,
      languages: getLocaleAlternates(""),
    },
    openGraph: {
      title: `${SITE_NAME} - ${t.homeTitle}`,
      url: `/${lang}`,
    },
  };
}

export default async function LocaleHomePage({ params }) {
  const { lang } = await params;
  if (!isValidLocale(lang)) {
    notFound();
  }
  const t = getLocaleStrings(lang);

  return (
    <main className="aq-home">
      <section className="aq-hero">
        <h1>{t.homeTitle}</h1>
        <p>{t.homeSubtitle}</p>
        <div className="aq-actions">
          <Link className="aq-btn aq-btn-primary" href={`/${lang}/docs`}>
            {t.quickStart}
          </Link>
          <Link className="aq-btn" href={`/${lang}/blog`}>
            {t.latestPosts}
          </Link>
          <Link className="aq-btn" href={`/${lang}/search`}>
            {t.searchTitle}
          </Link>
        </div>
      </section>

      <section className="aq-grid">
        <article className="aq-card">
          <h3>{t.docsTitle}</h3>
          <p>
            {lang === "zh"
              ? "完整 API、错误契约、迁移指南、工具与 Agent 用法。"
              : "Complete API guides, error contracts, migration notes, and tool/agent workflows."}
          </p>
        </article>
        <article className="aq-card">
          <h3>{t.blogTitle}</h3>
          <p>
            {lang === "zh"
              ? "发布记录、设计决策与最佳实践沉淀。"
              : "Release notes, design decisions, and production best practices."}
          </p>
        </article>
        <article className="aq-card">
          <h3>{t.searchTitle}</h3>
          <p>
            {lang === "zh"
              ? "内置搜索 + Pagefind 全文检索双引擎。"
              : "Dual search: built-in local search plus Pagefind full-text indexing."}
          </p>
        </article>
      </section>
    </main>
  );
}
