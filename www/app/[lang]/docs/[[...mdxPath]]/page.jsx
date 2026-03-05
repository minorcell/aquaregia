import { importPage } from "nextra/pages";
import { useMDXComponents as getDocsMDXComponents } from "nextra-theme-docs";
import { notFound } from "next/navigation";
import { generateSectionStaticParams } from "../../../../lib/nextra-helpers";
import {
  getLocaleAlternates,
  isValidLocale,
} from "../../../../lib/site-config";

const Wrapper = getDocsMDXComponents().wrapper;

function ensurePath(value) {
  return Array.isArray(value) ? value : [];
}

export async function generateStaticParams() {
  return generateSectionStaticParams("docs");
}

export async function generateMetadata({ params }) {
  const { lang, mdxPath } = await params;
  if (!isValidLocale(lang)) {
    return {};
  }

  const pathSegments = ["docs", ...ensurePath(mdxPath)];
  const { metadata } = await importPage(pathSegments, lang);
  const suffix = pathSegments.slice(1).join("/");

  return {
    ...metadata,
    alternates: {
      canonical: `/${lang}/docs/${suffix}`.replace(/\/$/, ""),
      languages: getLocaleAlternates(`/docs/${suffix}`.replace(/\/$/, "")),
    },
  };
}

export default async function DocsPage(props) {
  const params = await props.params;
  const { lang } = params;
  if (!isValidLocale(lang)) {
    notFound();
  }

  const mdxPath = ensurePath(params.mdxPath);
  const {
    default: MDXContent,
    toc,
    metadata,
    sourceCode,
  } = await importPage(["docs", ...mdxPath], lang);

  return (
    <Wrapper toc={toc} metadata={metadata} sourceCode={sourceCode}>
      <MDXContent {...props} params={{ ...params, mdxPath }} />
    </Wrapper>
  );
}
