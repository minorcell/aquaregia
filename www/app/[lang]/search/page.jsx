import { Search } from "nextra/components";
import { notFound } from "next/navigation";
import PagefindSearch from "../../../components/pagefind-search";
import { isValidLocale } from "../../../lib/site-config";

export default async function SearchPage({ params }) {
  const { lang } = await params;
  if (!isValidLocale(lang)) {
    notFound();
  }

  return (
    <main className="aq-home">
      <section className="aq-hero">
        <h1>{lang === "zh" ? "搜索" : "Search"}</h1>
        <p>
          {lang === "zh"
            ? "同时提供 Nextra 内置检索与 Pagefind 全文检索。"
            : "This page provides both Nextra local search and Pagefind full-text search."}
        </p>
      </section>

      <section className="aq-card" style={{ marginTop: "1rem" }}>
        <h2 style={{ marginTop: 0 }}>Nextra Search</h2>
        <Search placeholder={lang === "zh" ? "搜索文档..." : "Search docs..."} />
      </section>

      <PagefindSearch
        locale={lang}
        placeholder={lang === "zh" ? "输入关键词进行全文搜索" : "Type to full-text search"}
        emptyText={lang === "zh" ? "输入关键词开始搜索。" : "Type a keyword to start searching."}
        noResultText={lang === "zh" ? "没有结果。" : "No results found."}
      />
    </main>
  );
}
