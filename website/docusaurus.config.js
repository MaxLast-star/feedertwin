// @ts-check

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'FeederTwin',
  tagline: 'Имитационная модель и исследование стратегий управления системой поштучной подачи',
  favicon: 'img/favicon.svg',
  url: 'https://maxlast-star.github.io',
  baseUrl: '/feedertwin/',
  organizationName: 'MaxLast-star',
  projectName: 'feedertwin',
  onBrokenLinks: 'warn',
  markdown: {
    mermaid: true,
    hooks: { onBrokenMarkdownLinks: 'warn' },
  },
  themes: ['@docusaurus/theme-mermaid'],
  i18n: { defaultLocale: 'ru', locales: ['ru'] },
  presets: [
    ['classic', /** @type {import('@docusaurus/preset-classic').Options} */ ({
      docs: {
        path: '../docs/sa',
        routeBasePath: 'sa',
        sidebarPath: require.resolve('./sidebars.js'),
        editUrl: 'https://github.com/MaxLast-star/feedertwin/tree/main/',
      },
      blog: false,
      theme: { customCss: require.resolve('./src/css/custom.css') },
    })],
  ],
  themeConfig: /** @type {import('@docusaurus/preset-classic').ThemeConfig} */ ({
    // Палитра Mermaid согласована с дизайном «инженерный чертёж».
    mermaid: {
      theme: { light: 'neutral', dark: 'dark' },
      options: {
        themeVariables: {
          primaryColor: '#eaf0f6',
          primaryBorderColor: '#1d3a5f',
          primaryTextColor: '#1d3a5f',
          lineColor: '#41608a',
          fontFamily: "'IBM Plex Mono', ui-monospace, monospace",
        },
      },
    },
    navbar: {
      title: 'FeederTwin',
      items: [
        { to: '/sa/01-concept', label: 'SA-документация', position: 'left' },
        { to: '/dashboard', label: 'What-if дашборд', position: 'left' },
        { href: 'https://github.com/MaxLast-star/feedertwin', label: 'GitHub', position: 'right' },
      ],
    },
    footer: {
      style: 'dark',
      copyright: 'FeederTwin · Ластовенко М.А. · НИТУ МИСИС · MIT License',
    },
  }),
};
module.exports = config;
