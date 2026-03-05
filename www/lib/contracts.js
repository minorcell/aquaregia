import { z } from "zod";

export const DocFrontmatterSchema = z
  .object({
    title: z.string(),
    description: z.string(),
    searchable: z.boolean().optional(),
    tags: z.array(z.string()).optional(),
  })
  .passthrough();

export const BlogFrontmatterSchema = z
  .object({
    title: z.string(),
    description: z.string(),
    date: z.union([z.string(), z.date()]).transform((value) =>
      typeof value === "string" ? value : value.toISOString(),
    ),
    author: z.string().default("Aquaregia Team"),
    tags: z.array(z.string()).default([]),
  })
  .passthrough();

export const ApiPageFrontmatterSchema = DocFrontmatterSchema.extend({
  crate_version: z.string(),
  stability: z.enum(["stable", "experimental", "deprecated"]),
  since: z.string(),
}).passthrough();

export const SyncManifestEntrySchema = z.object({
  source: z.string(),
  targetLanguage: z.enum(["zh", "en"]),
  targetPath: z.string(),
  strategy: z.enum(["replace", "append"]).default("replace"),
});

export const SyncManifestSchema = z.array(SyncManifestEntrySchema);

export const ComponentDemoSchema = z.object({
  component: z.string(),
  docPath: z.string(),
  description: z.string(),
});

export const ComponentDemoIndexSchema = z.array(ComponentDemoSchema);

export const componentDemoIndex = ComponentDemoIndexSchema.parse([
  {
    component: "Callout",
    docPath: "/docs/demos/components",
    description: "Highlight notes, warnings, and tips",
  },
  {
    component: "Tabs",
    docPath: "/docs/demos/components",
    description: "Switch between installation variants",
  },
  {
    component: "Cards",
    docPath: "/docs/demos/components",
    description: "Grid links for quick navigation",
  },
  {
    component: "Steps",
    docPath: "/docs/demos/components",
    description: "Ordered workflow rendering",
  },
  {
    component: "FileTree",
    docPath: "/docs/demos/components",
    description: "Show project layout visually",
  },
  {
    component: "Mermaid",
    docPath: "/docs/demos/components",
    description: "Render diagrams in Markdown",
  },
  {
    component: "MathJax",
    docPath: "/docs/demos/components",
    description: "Render formulas in docs",
  },
]);
