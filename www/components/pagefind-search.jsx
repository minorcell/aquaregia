"use client";

import { useEffect, useMemo, useState } from "react";

async function ensurePagefind() {
  if (typeof window === "undefined") {
    return null;
  }
  if (window.pagefind) {
    return window.pagefind;
  }

  await new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "/_pagefind/pagefind.js";
    script.async = true;
    script.onload = resolve;
    script.onerror = reject;
    document.body.appendChild(script);
  });

  return window.pagefind || null;
}

export default function PagefindSearch({
  locale,
  placeholder,
  emptyText,
  noResultText,
}) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [status, setStatus] = useState("idle");

  useEffect(() => {
    let cancelled = false;

    async function runSearch() {
      if (!query.trim()) {
        setResults([]);
        setStatus("idle");
        return;
      }

      setStatus("loading");

      try {
        const pagefind = await ensurePagefind();
        if (!pagefind) {
          if (!cancelled) {
            setResults([]);
            setStatus("error");
          }
          return;
        }

        const search = await pagefind.search(query);

        const loaded = await Promise.all(search.results.map((item) => item.data()));
        const scoped = loaded.filter((item) => item.url.includes(`/${locale}/`));
        if (cancelled) {
          return;
        }

        setResults(scoped);
        setStatus("done");
      } catch {
        if (!cancelled) {
          setResults([]);
          setStatus("error");
        }
      }
    }

    runSearch();

    return () => {
      cancelled = true;
    };
  }, [locale, query]);

  const content = useMemo(() => {
    if (status === "idle") {
      return <p className="pf-hint">{emptyText}</p>;
    }
    if (status === "loading") {
      return <p className="pf-hint">Loading...</p>;
    }
    if (status === "error") {
      return <p className="pf-hint">Pagefind index is unavailable.</p>;
    }
    if (!results.length) {
      return <p className="pf-hint">{noResultText}</p>;
    }

    return (
      <ul className="pf-results">
        {results.map((item) => (
          <li key={item.url} className="pf-item">
            <a href={item.url} className="pf-title">
              {item.meta.title || item.url}
            </a>
            {item.excerpt ? (
              <p
                className="pf-excerpt"
                dangerouslySetInnerHTML={{ __html: item.excerpt }}
              />
            ) : null}
          </li>
        ))}
      </ul>
    );
  }, [emptyText, noResultText, results, status]);

  return (
    <section className="pagefind-panel">
      <label className="pf-label" htmlFor="pagefind-search-input">
        Pagefind
      </label>
      <input
        id="pagefind-search-input"
        type="search"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        placeholder={placeholder}
        className="pf-input"
      />
      {content}
    </section>
  );
}
