import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import { appName, docsRoute, gitConfig } from './shared';

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: appName,
    },
    links: [
      {
        text: 'Docs',
        url: docsRoute,
        active: 'url',
      },
      {
        text: 'Examples',
        url: `https://github.com/${gitConfig.user}/${gitConfig.repo}/tree/${gitConfig.branch}/examples`,
        active: 'none',
      },
      {
        text: 'API Reference',
        url: 'https://docs.rs/aquaregia',
        active: 'none',
      },
    ],
    githubUrl: `https://github.com/${gitConfig.user}/${gitConfig.repo}`,
  };
}
