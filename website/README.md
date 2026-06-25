# Сайт SA-документации FeederTwin

Docusaurus-сайт собирается из Markdown-документов в `../docs/sa/`
и встраивает интерактивный what-if дашборд.

## Локальная сборка

```bash
cd website
npm install
```

Перед сборкой вложите дашборд в статику сайта (Windows PowerShell):

```powershell
New-Item -ItemType Directory -Force static\dashboard | Out-Null
Copy-Item ..\dashboard\index.html static\dashboard\index.html
```

Затем:

```bash
npm run build      # статика в website/build
npm run serve      # локальный просмотр
```

Документы (`docs/sa/*.md`) читаются и без сборки — прямо на GitHub.
Каталоги `node_modules`, `build`, `static/dashboard` не хранятся в репозитории.

## Публикация на GitHub Pages

Settings → Pages → Source: GitHub Actions, либо ручной деплой ветки
`gh-pages`. baseUrl уже настроен на `/feedertwin/`.
