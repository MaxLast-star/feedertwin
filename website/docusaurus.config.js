// @ts-check
const config = {
  title: 'FeederTwin',
  tagline: 'Имитационная модель и исследование стратегий управления системой поштучной подачи',
  favicon: 'img/favicon.svg',
  url: 'https://maxlast-star.github.io',
  baseUrl: '/feedertwin/',
  organizationName: 'MaxLast-star',
  projectName: 'feedertwin',
  onBrokenLinks: 'warn',
  markdown: { hooks: { onBrokenMarkdownLinks: 'warn' } },
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
