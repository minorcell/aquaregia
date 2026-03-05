import { Banner, Search } from "nextra/components";
import { Footer, Layout as DocsLayout, Navbar } from "nextra-theme-docs";
import { getPageMap } from "nextra/page-map";
import { notFound } from "next/navigation";
import {
  DOCS_REPOSITORY_BASE,
  ISSUE_URL,
  LOCALE_LABELS,
  LOCALES,
  REPOSITORY_URL,
  SITE_NAME,
  getLocaleStrings,
  isValidLocale,
} from "../../../lib/site-config";

export default async function DocsRouteLayout({ children, params }) {
  const { lang } = await params;
  if (!isValidLocale(lang)) {
    notFound();
  }

  const t = getLocaleStrings(lang);
  const pageMap = await getPageMap(`/${lang}/docs`);

  const banner = (
    <Banner storageKey="aquaregia-docs-banner">
      {lang === "zh"
        ? "Aquaregia v2 API 已上线，文档已同步更新"
        : "Aquaregia v2 API is live with updated documentation"}
    </Banner>
  );

  const navbar = (
    <Navbar logo={<strong>{SITE_NAME}</strong>} projectLink={REPOSITORY_URL}>
      <a href={`/${lang}/blog`}>{t.blogTitle}</a>
      <a href={`/${lang}/search`}>{t.searchTitle}</a>
    </Navbar>
  );

  const footer = (
    <Footer>
      MIT {new Date().getFullYear()} © Aquaregia. {lang === "zh" ? "贡献者共建" : "Built by contributors."}
    </Footer>
  );

  return (
    <DocsLayout
      banner={banner}
      navbar={navbar}
      footer={footer}
      pageMap={pageMap}
      docsRepositoryBase={DOCS_REPOSITORY_BASE}
      editLink={t.editThisPage}
      feedback={{
        content: t.feedback,
        labels: "feedback,docs",
        link: `${ISSUE_URL}?labels=feedback,docs`,
      }}
      i18n={LOCALES.map((locale) => ({
        locale,
        name: LOCALE_LABELS[locale],
      }))}
      search={<Search placeholder={lang === "zh" ? "搜索文档..." : "Search docs..."} />}
      toc={{ title: t.tocTitle, backToTop: t.tocBackToTop }}
      sidebar={{
        defaultOpen: true,
        defaultMenuCollapseLevel: 2,
        toggleButton: true,
      }}
      navigation
      darkMode
      themeSwitch={{
        light: lang === "zh" ? "浅色" : "Light",
        dark: lang === "zh" ? "深色" : "Dark",
        system: lang === "zh" ? "跟随系统" : "System",
      }}
    >
      {children}
    </DocsLayout>
  );
}
