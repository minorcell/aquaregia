import nextra from "nextra";

const withNextra = nextra({
  contentDirBasePath: "/",
  defaultShowCopyCode: true,
  search: {
    codeblocks: true,
  },
  readingTime: true,
  latex: {
    renderer: "katex",
  },
  unstable_shouldAddLocaleToLinks: true,
});

export default withNextra({
  i18n: {
    locales: ["zh", "en"],
    defaultLocale: "zh",
  },
  reactStrictMode: true,
});
