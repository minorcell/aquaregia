import Image from 'next/image';
import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import { appName, docsRoute, gitConfig } from './shared';

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: (
        <span className="inline-flex items-center gap-2 font-semibold">
          <Image src="/brand/aquaregia-logo.svg" alt="" width={20} height={20} className="shrink-0" />
          {appName}
        </span>
      ),
    },
    links: [
      {
        text: 'Overview',
        url: docsRoute,
        active: 'url',
      },
    ],
    githubUrl: `https://github.com/${gitConfig.user}/${gitConfig.repo}`,
  };
}

export function docsSidebarFooter() {
  const links = [
    {
      text: 'Examples',
      url: `https://github.com/${gitConfig.user}/${gitConfig.repo}/tree/${gitConfig.branch}/examples`,
    },
    {
      text: 'API Reference',
      url: 'https://docs.rs/aquaregia',
    },
  ];

  return (
    <div className="mt-3 flex flex-col gap-1 border-t pt-3 text-sm">
      {links.map((link) => (
        <a
          key={link.text}
          href={link.url}
          className="rounded-md px-2 py-1.5 text-fd-muted-foreground transition-colors hover:bg-fd-accent/50 hover:text-fd-accent-foreground"
        >
          {link.text}
        </a>
      ))}
    </div>
  );
}
