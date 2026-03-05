import { importPage } from "nextra/pages";
import { notFound } from "next/navigation";
import { generateSectionStaticParams } from "../../../../lib/nextra-helpers";
import {
  getLocaleAlternates,
  isValidLocale,
} from "../../../../lib/site-config";

function ensurePath(value) {
  return Array.isArray(value) ? value : [];
}

export async function generateStaticParams() {
  const all = await generateSectionStaticParams("blog", { includeIndex: false });
  return all.filter(({ mdxPath }) => mdxPath.length > 0 && mdxPath[0] !== "tags");
}

export async function generateMetadata({ params }) {
  const { lang, mdxPath } = await params;
  if (!isValidLocale(lang)) {
    return {};
  }

  const pathSegments = ["blog", ...ensurePath(mdxPath)];
  const { metadata } = await importPage(pathSegments, lang);
  const suffix = pathSegments.slice(1).join("/");

  return {
    ...metadata,
    alternates: {
      canonical: `/${lang}/blog/${suffix}`,
      languages: getLocaleAlternates(`/blog/${suffix}`),
    },
  };
}

export default async function BlogMdxPage(props) {
  const params = await props.params;
  const { lang } = params;
  if (!isValidLocale(lang)) {
    notFound();
  }

  const mdxPath = ensurePath(params.mdxPath);
  if (!mdxPath.length) {
    notFound();
  }

  const { default: MDXContent } = await importPage(["blog", ...mdxPath], lang);
  return <MDXContent {...props} params={{ ...params, mdxPath }} />;
}
