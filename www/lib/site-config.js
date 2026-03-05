export const SITE_NAME = "Aquaregia Docs";
export const SITE_DESCRIPTION =
  "Aquaregia documentation site for Rust LLM client and agent APIs.";
export const SITE_URL =
  process.env.NEXT_PUBLIC_SITE_URL || "https://aquaregia-docs.vercel.app";

export const REPOSITORY_URL = "https://github.com/mcell/aquaregia";
export const DOCS_REPOSITORY_BASE = `${REPOSITORY_URL}/tree/main`;
export const ISSUE_URL = `${REPOSITORY_URL}/issues/new`;

export const DEFAULT_LOCALE = "zh";
export const LOCALES = ["zh", "en"];

export const LOCALE_LABELS = {
  zh: "简体中文",
  en: "English",
};

export const SITE_STRINGS = {
  zh: {
    docsTitle: "文档",
    blogTitle: "博客",
    searchTitle: "搜索",
    homeTitle: "Aquaregia 文档站",
    homeSubtitle: "面向 Rust 的 LLM Client 与 Agent SDK",
    quickStart: "快速开始",
    latestPosts: "最新文章",
    feedback: "有问题？给我们反馈",
    editThisPage: "编辑此页面",
    tocTitle: "本页目录",
    tocBackToTop: "回到顶部",
    readMore: "继续阅读 ->",
  },
  en: {
    docsTitle: "Docs",
    blogTitle: "Blog",
    searchTitle: "Search",
    homeTitle: "Aquaregia Docs",
    homeSubtitle: "LLM Client and Agent SDK for Rust",
    quickStart: "Quick Start",
    latestPosts: "Latest Posts",
    feedback: "Questions? Give us feedback",
    editThisPage: "Edit this page",
    tocTitle: "On This Page",
    tocBackToTop: "Back to top",
    readMore: "Read more ->",
  },
};

export function isValidLocale(lang) {
  return LOCALES.includes(lang);
}

export function getLocaleStrings(lang) {
  return SITE_STRINGS[lang] || SITE_STRINGS[DEFAULT_LOCALE];
}

export function getLocaleAlternates(pathname) {
  return Object.fromEntries(
    LOCALES.map((lang) => [lang, `/${lang}${pathname}`.replace(/\/+/g, "/")]),
  );
}
