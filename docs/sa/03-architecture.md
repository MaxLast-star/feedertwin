---
id: 03-architecture
title: 3. Архитектура (C4)
sidebar_position: 3
---

# Архитектура модели

Описание по нотации C4: контекст → контейнеры → компоненты.

## Уровень 1. Контекст

```mermaid
flowchart TD
    user["👤 Пользователь<br/>(конструктор / аналитик)"]
    nir[("Параметры НИР<br/>A, f, бюджет времени,<br/>целевая производительность")]
    sys["<b>FeederTwin</b><br/>вычислительная модель<br/>системы подачи"]
    art["Артефакты:<br/>отчёты · дашборд · SA-документы"]

    user -->|параметры, сценарии| sys
    nir -->|калибровка| sys
    sys -->|метрики, графики,<br/>рекомендации| art
    art -->|выводы| user

    classDef system fill:#1d3a5f,stroke:#1d3a5f,color:#f6f7f4;
    classDef ext fill:#eaf0f6,stroke:#41608a,color:#1d3a5f;
    class sys system;
    class user,nir,art ext;
```

Внешний источник данных — параметры из НИР (амплитуда, частота, бюджет
времени контура, целевая производительность).

## Уровень 2. Контейнеры

| Контейнер | Технология | Назначение |
|---|---|---|
| Расчётное ядро | Python ≥ 3.11, NumPy, SimPy | модели и алгоритмы |
| Сценарии экспериментов | Python + Matplotlib | прогоны, графики, отчёты |
| What-if дашборд | статический HTML + Plotly | интерактивное представление |
| Документация | Markdown / Docusaurus | SA-комплект и сайт |
| CI | GitHub Actions | автоматические тесты и линт |

## Уровень 3. Компоненты расчётного ядра

```mermaid
flowchart TD
    transport["<b>transport.py</b><br/>механика вибротранспортирования"]
    flowsim["<b>flowsim.py</b><br/>поток деталей + контур контроля (SimPy)"]
    strategies["<b>strategies.py</b><br/>Baseline · Adaptive · Metering"]
    mcda["<b>mcda.py</b><br/>AHP + TOPSIS + чувствительность"]

    transport -->|"v(A,f,α,μ), режим Γ →<br/>время прохода зоны"| flowsim
    strategies -->|"хуки: on_frame, rate_factor,<br/>on_admission, on_engagement"| flowsim
    flowsim -->|"метрики Monte Carlo"| mcda

    classDef mod fill:#eaf0f6,stroke:#1d3a5f,color:#1d3a5f;
    class transport,flowsim,strategies,mcda mod;
```

### Интерфейс стратегии управления

Точки, в которых имитационное ядро опрашивает стратегию:

| Хук | Когда вызывается | Что определяет |
|---|---|---|
| `on_frame(now, positive)` | каждый кадр детектора | вводить ли заслонку |
| `rate_factor(now)` | при генерации прихода | множитель интенсивности вибрации |
| `on_admission(now, rng)` | при входе детали в зону | дозирование (момент след. входа) |
| `on_engagement(now)` | при вводе заслонки | обратная связь стратегии |

Такое разделение обеспечивает требование FR-5 (сменные стратегии) и
гарантирует обратную совместимость: базовая стратегия воспроизводит
поведение исходной модели бит-в-бит (тест `test_baseline_strategy_is_backward_compatible`).

## Поток данных эксперимента

```mermaid
flowchart TD
    tp["TrayParams"] --> mv["transport.mean_velocity"]
    mv --> tz["время прохода зоны"]
    fp["FlowParams + Strategy"] --> rr["flowsim.run_replications"]
    tz --> rr
    rr --> met["метрики (Monte Carlo)"]
    met --> mcda["mcda.ahp_weights + topsis"]
    mcda --> rank["ранжирование альтернатив"]
    met --> out["дашборд · отчёты · SA-документы"]
    rank --> out

    classDef data fill:#fff,stroke:#41608a,color:#1d3a5f;
    classDef proc fill:#eaf0f6,stroke:#1d3a5f,color:#1d3a5f;
    class tp,fp,tz,met,rank data;
    class mv,rr,mcda,out proc;
```
