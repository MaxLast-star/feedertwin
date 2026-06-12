"""Сценарий многокритериального выбора (этап 4).

Запуск:  python scripts/run_mcda_study.py

Два кейса:
1. Выбор стратегии управления — альтернативы и метрики из этапа 3
   (реальные результаты Monte Carlo при 5 дет/с), экспертные критерии
   сложности; веса критериев — AHP с проверкой согласованности;
   ранжирование — TOPSIS; анализ чувствительности веса «качество подачи».
2. Выбор метода контроля единичности — формализация сравнительного
   анализа из главы 4 НИР (однолучевой датчик / световой барьер /
   машинное зрение); анализ чувствительности веса «стоимость».
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from feedertwin.mcda import ahp_weights, topsis, topsis_weight_sweep  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"

# ============================================================
# Кейс 1. Выбор стратегии управления (данные этапа 3, 5 дет/с)
# ============================================================
STRAT_NAMES = ["Без контроля", "Базовая пороговая", "Адаптивная вибрация", "Дозирующий выпуск"]
STRAT_CRITERIA = ["Качество подачи\n(дефекты/1000)", "Производительность\n(дет/с)",
                  "Сложность реализации\n(балл 1–5)", "Износ механизма\n(duty, %)"]
# Дефекты и производительность — Monte Carlo этапа 3; сложность — экспертная
# (датчики/логика/калибровка); износ — доля времени работы заслонки.
STRAT_X = np.array([
    #  деф./1000  дет/с  сложн.  duty,%
    [100.9,      5.37,  1.0,    0.0],   # без контроля
    [95.9,       5.33,  2.0,    6.6],   # базовая пороговая
    [57.3,       3.64,  4.0,    2.9],   # адаптивная вибрация
    [1.4,        5.18,  4.0,    4.6],   # дозирующий выпуск
])
STRAT_BENEFIT = [False, True, False, False]

# Парные сравнения критериев (шкала Саати): качество подачи — главный
# критерий (сама цель системы), затем производительность, затем сложность,
# затем износ.
STRAT_PAIRWISE = np.array([
    [1.0, 3.0, 5.0, 7.0],
    [1 / 3, 1.0, 3.0, 5.0],
    [1 / 5, 1 / 3, 1.0, 3.0],
    [1 / 7, 1 / 5, 1 / 3, 1.0],
])

# ============================================================
# Кейс 2. Выбор метода контроля единичности (глава 4 НИР)
# ============================================================
SENSOR_NAMES = ["Однолучевой датчик", "Световой барьер", "Машинное зрение (CV)"]
SENSOR_CRITERIA = ["Стоимость\n(балл 1–5)", "Информативность\n(балл 1–5)",
                   "Сложность внедрения\n(балл 1–5)", "Гибкость к номенклатуре\n(балл 1–5)"]
# Балльные оценки 1–5 формализуют качественное сравнение главы 4 НИР.
SENSOR_X = np.array([
    # стоим.  информ.  сложн.  гибк.
    [1.0,    1.0,     1.0,    1.0],   # однолучевой
    [2.0,    2.0,     2.0,    2.0],   # барьер
    [4.0,    5.0,     4.0,    5.0],   # машинное зрение
])
SENSOR_BENEFIT = [False, True, False, True]

# Информативность важнее всего (различение «сдвоек», а не только факта
# детали), затем гибкость, затем стоимость, затем сложность внедрения.
SENSOR_PAIRWISE = np.array([
    [1.0, 1 / 4, 2.0, 1 / 3],
    [4.0, 1.0, 5.0, 2.0],
    [1 / 2, 1 / 5, 1.0, 1 / 4],
    [3.0, 1 / 2, 4.0, 1.0],
])


def run_case(title: str, names: list[str], criteria: list[str],
             x: np.ndarray, benefit: list[bool], pairwise: np.ndarray):
    """Полный расчёт кейса: AHP → TOPSIS, печать таблиц."""
    print("-" * 72)
    print(title)
    print("-" * 72)
    ahp = ahp_weights(pairwise)
    crit_flat = [c.replace("\n", " ") for c in criteria]
    print(f"AHP: λ_max = {ahp.lambda_max:.3f}, CI = {ahp.ci:.3f}, CR = {ahp.cr:.3f} "
          f"({'согласована' if ahp.is_consistent else 'НЕ согласована!'})")
    for c, w in zip(crit_flat, ahp.weights, strict=True):
        print(f"  вес {c}: {w:.3f}")
    t = topsis(x, ahp.weights, benefit)
    print("TOPSIS:")
    for i in t.ranking:
        print(f"  {t.closeness[i]:.3f}  {names[i]}")
    return ahp, t


def fig_closeness(t_strat, t_sensor) -> None:
    """Близости TOPSIS для обоих кейсов."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, t, names, title in (
        (axes[0], t_strat, STRAT_NAMES, "Кейс 1. Выбор стратегии управления"),
        (axes[1], t_sensor, SENSOR_NAMES, "Кейс 2. Выбор метода контроля (НИР)"),
    ):
        order = t.ranking[::-1]
        colors = ["tab:blue" if i == t.ranking[0] else "0.6" for i in order]
        ax.barh([names[i] for i in order], [t.closeness[i] for i in order], color=colors)
        ax.set_xlabel("Близость к идеальному решению (TOPSIS)")
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="x", alpha=0.3)
    fig.suptitle("Многокритериальный выбор: AHP-веса + TOPSIS", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mcda_closeness.png", dpi=150)
    plt.close(fig)


def fig_sensitivity(ahp_strat, ahp_sensor) -> None:
    """Чувствительность ранжирования к весам ключевых критериев."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    grid, cl = topsis_weight_sweep(STRAT_X, ahp_strat.weights, STRAT_BENEFIT, 0)
    for j, name in enumerate(STRAT_NAMES):
        axes[0].plot(grid, cl[:, j], label=name, lw=1.8)
    axes[0].axvline(ahp_strat.weights[0], ls="--", color="0.3", lw=1.2)
    axes[0].annotate("вес AHP", xy=(ahp_strat.weights[0] + 0.01, 0.05), fontsize=8)
    axes[0].set_xlabel("Вес критерия «качество подачи»")
    axes[0].set_title("Кейс 1: чувствительность к весу качества", fontsize=10)

    grid2, cl2 = topsis_weight_sweep(SENSOR_X, ahp_sensor.weights, SENSOR_BENEFIT, 0)
    for j, name in enumerate(SENSOR_NAMES):
        axes[1].plot(grid2, cl2[:, j], label=name, lw=1.8)
    axes[1].axvline(ahp_sensor.weights[0], ls="--", color="0.3", lw=1.2)
    axes[1].annotate("вес AHP", xy=(ahp_sensor.weights[0] + 0.01, 0.05), fontsize=8)
    axes[1].set_xlabel("Вес критерия «стоимость»")
    axes[1].set_title("Кейс 2: чувствительность к весу стоимости", fontsize=10)

    for ax in axes:
        ax.set_ylabel("Близость TOPSIS")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Анализ чувствительности весов (поиск точек смены ранга)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mcda_sensitivity.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("FeederTwin · Этап 4 · Многокритериальный анализ (AHP + TOPSIS)")
    print("=" * 72)
    ahp1, t1 = run_case("Кейс 1. Выбор стратегии управления (метрики этапа 3, 5 дет/с)",
                        STRAT_NAMES, STRAT_CRITERIA, STRAT_X, STRAT_BENEFIT, STRAT_PAIRWISE)
    ahp2, t2 = run_case("Кейс 2. Выбор метода контроля единичности (глава 4 НИР)",
                        SENSOR_NAMES, SENSOR_CRITERIA, SENSOR_X, SENSOR_BENEFIT, SENSOR_PAIRWISE)
    fig_closeness(t1, t2)
    print("  reports/figures/mcda_closeness.png")
    fig_sensitivity(ahp1, ahp2)
    print("  reports/figures/mcda_sensitivity.png")
    print("Готово.")


if __name__ == "__main__":
    main()
