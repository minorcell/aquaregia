import { source } from '@/lib/source';
import { DocsLayout } from 'fumadocs-ui/layouts/docs';
import { baseOptions, docsSidebarFooter } from '@/lib/layout.shared';

export default function Layout({ children }: LayoutProps<'/docs'>) {
  return (
    <DocsLayout
      tree={source.getPageTree()}
      sidebar={{
        footer: docsSidebarFooter(),
      }}
      {...baseOptions()}
    >
      {children}
    </DocsLayout>
  );
}
