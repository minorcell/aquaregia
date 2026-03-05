import { generateStaticParamsFor } from "nextra/pages";

const generateAllStaticParams = generateStaticParamsFor("mdxPath", "lang");

function ensureArray(value) {
  return Array.isArray(value) ? value : [];
}

export async function generateSectionStaticParams(section, options = {}) {
  const { includeIndex = true } = options;
  const all = await generateAllStaticParams();

  return all
    .filter(({ mdxPath }) => ensureArray(mdxPath)[0] === section)
    .map(({ lang, mdxPath }) => ({
      lang,
      mdxPath: ensureArray(mdxPath).slice(1),
    }))
    .filter(({ mdxPath }) => includeIndex || mdxPath.length > 0);
}
