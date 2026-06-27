# Aquaregia Docs

This directory contains the Aquaregia documentation site.

## Development

```bash
pnpm install
pnpm dev
```

Open http://localhost:3000.

## Content

- Landing page: `src/app/(home)/page.tsx`
- Documentation pages: `content/docs/*.mdx`
- Navigation order: `content/docs/meta.json`
- Shared docs layout: `src/lib/layout.shared.tsx`

Write docs as current API documentation. Avoid migration notes in the guide pages; put release history in the root `CHANGELOG.md`.

## Checks

```bash
pnpm types:check
pnpm build
```
