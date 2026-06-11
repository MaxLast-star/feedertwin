"""Сценарий сравнения стратегий управления (этап 3).

Запуск:  python scripts/run_strategies_study.py

Три эксперимента Monte Carlo (10 репликаций × 300 с):
1. Кривые «дефекты от интенсивности» для четырёх вариантов управления.
2. Парето-фронт «производительность ↔ дефекты подачи».
3. Переходный процесс адаптивной стратегии при перегрузке (трасса
   множителя интенсивности вибрации).
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from feedertwin.flowsim import FlowParams, run_replications, run_simulation  # noqa: E402
from feedertwin.strategies import (  # noqa: E402
    AdaptiveVibration,
    BaselineThreshold,
    MeteringGate,
)

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
BASE = FlowParams(sim_time_s=300.0, seed=500)
N_REPS = 10
RATES = np.arange(3.0, 8.1, 1.0)

STRATEGIES = (
    ("без контроля", lambda: BaselineThreshold(n_confirm_frames=10**9), "tab:gray", "o"),
    ("базовая пороговая (НИР)", lambda: None, "tab:green", "s"),
    ("адаптивная вибрация", AdaptiveVibration, "tab:orange", "^"),
    ("дозирующий выпуск", MeteringGate, "tab:blue", "D"),
)


def collect() -> dict[str, dict[str, list[float]]]:
    """Прогоны всех стратегий по сетке интенсивностей."""
    data: dict[str, dict[str, list[float]]] = {}
    for name, factory, _, _ in STRATEGIES:
        d = {"esc": [], "esc_h": [], "thr": [], "thr_h": []}
        for r in RATES:
            s = run_replications(
                replace(BASE, arrival_rate_hz=float(r)), n_reps=N_REPS, strategy=factory()
            )
            d["esc"].append(s.escaped_per_1000[0])
            d["esc_h"].append(s.escaped_per_1000[1])
            d["thr"].append(s.throughput_hz[0])
            d["thr_h"].append(s.throughput_hz[1])
        data[name] = d
    return data


def fig_strategy_capacity(data: dict) -> None:
    """Дефекты от интенсивности потока для каждой стратегии."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, _, color, marker in STRATEGIES:
        d = data[name]
        ax.errorbar(RATES, d["esc"], yerr=d["esc_h"], marker=marker, ms=4,
                    capsize=3, color=color, label=name)
    ax.axvline(5.0, ls="--", color="0.3", lw=1.2)
    ax.annotate("цель НИР: 5 дет/с", xy=(5.05, ax.get_ylim()[1] * 0.9), fontsize=9)
    ax.set_xlabel("Интенсивность потока деталей, дет/с")
    ax.set_ylabel("Дефекты подачи, на 1000 деталей")
    ax.set_title("Стратегии управления: дефекты от интенсивности\n(Monte Carlo, 95% ДИ)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "strategies_capacity.png", dpi=150)
    plt.close(fig)


def fig_pareto(data: dict) -> None:
    """Парето-фронт «производительность ↔ дефекты» по стратегиям и режимам."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, _, color, marker in STRATEGIES:
        d = data[name]
        ax.plot(d["thr"], d["esc"], marker=marker, ms=6, lw=1.2, color=color,
                label=name, alpha=0.9)
        for x, y, r in zip(d["thr"], d["esc"], RATES, strict=False):
            ax.annotate(f"{r:g}", xy=(x, y), xytext=(3, 3),
                        textcoords="offset points", fontsize=7, color=color)
    ax.set_xlabel("Производительность, дет/с")
    ax.set_ylabel("Дефекты подачи, на 1000 деталей")
    ax.set_title("Парето-пространство «производительность ↔ качество подачи»\n"
                 "(подписи точек — интенсивность потока, дет/с)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "strategies_pareto.png", dpi=150)
    plt.close(fig)


def fig_adaptive_trace() -> None:
    """Переходный процесс адаптивной стратегии при перегрузке 7 дет/с."""
    p = replace(BASE, arrival_rate_hz=7.0, seed=501)
    m = run_simulation(p, strategy=AdaptiveVibration())
    ts = [t for t, _ in m.control_trace] + [p.sim_time_s]
    fs = [f for _, f in m.control_trace] + [m.control_trace[-1][1]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.step(ts, fs, where="post", color="tab:orange", lw=1.6)
    ax.set_ylim(0.0, 1.1)
    ax.axhline(1.0, ls=":", color="0.5", lw=1)
    ax.axhline(0.5, ls=":", color="0.5", lw=1)
    ax.annotate("нижняя граница интенсивности (0,5)", xy=(2, 0.52), fontsize=8, color="0.4")
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Множитель интенсивности вибрации")
    ax.set_title(
        "Адаптивная стратегия при перегрузке (7 дет/с): контур\n"
        "«частота срабатываний → интенсивность подачи» в действии"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "strategies_adaptive_trace.png", dpi=150)
    plt.close(fig)


def print_summary(data: dict) -> None:
    """Сводка в рабочей точке 5 дет/с."""
    i = int(np.where(RATES == 5.0)[0][0])
    print("=" * 72)
    print("FeederTwin · Этап 3 · Сравнение стратегий управления (5 дет/с)")
    print("=" * 72)
    for name, _, _, _ in STRATEGIES:
        d = data[name]
        print(f"  {name:28s} дефекты {d['esc'][i]:7.1f} ± {d['esc_h'][i]:4.1f} /1000,"
              f"  производительность {d['thr'][i]:.2f} дет/с")
    print("=" * 72)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Эксперименты Monte Carlo (займёт ~3–5 минут)...")
    data = collect()
    print_summary(data)
    fig_strategy_capacity(data)
    print("  reports/figures/strategies_capacity.png")
    fig_pareto(data)
    print("  reports/figures/strategies_pareto.png")
    fig_adaptive_trace()
    print("  reports/figures/strategies_adaptive_trace.png")
    print("Готово.")


if __name__ == "__main__":
    main()
