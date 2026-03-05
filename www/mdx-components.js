import { useMDXComponents as getNextraMDXComponents } from "nextra/mdx-components";
import {
  Bleed,
  Button,
  Callout,
  Cards,
  Collapse,
  FileTree,
  Mermaid,
  MathJax,
  MathJaxContext,
  Playground,
  Select,
  Steps,
  Tabs,
} from "nextra/components";
import { useMDXComponents as getDocsMDXComponents } from "nextra-theme-docs";
import { useMDXComponents as getBlogMDXComponents } from "nextra-theme-blog";

const { wrapper: _docsWrapper, ...docsComponents } = getDocsMDXComponents();
const { wrapper: _blogWrapper, ...blogComponents } = getBlogMDXComponents();

export function useMDXComponents(components) {
  return {
    ...getNextraMDXComponents(),
    ...blogComponents,
    ...docsComponents,
    Bleed,
    Button,
    Callout,
    Cards,
    Collapse,
    FileTree,
    Mermaid,
    MathJax,
    MathJaxContext,
    Playground,
    Select,
    Steps,
    Tabs,
    ...components,
  };
}
