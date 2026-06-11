"""Сценарий исследования контура контроля единичной подачи (этап 2).

Запуск:  python scripts/run_flow_study.py

Три эксперимента Monte Carlo (12 репликаций × 300 с модельного времени):
1. Свип порога детектора — компромисс «дефекты ↔ ложные срабатывания».
2. Кривая производительности — дефекты от интенсивности потока,
   с контролем и без.
3. Обратная задача: какое качество детектора минимально необходимо.
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

from feedertwin.flowsim import FlowParams, run_replications  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
BASE = FlowParams(sim_time_s=300.0, seed=100)
N_REPS = 12
NIR_THRESHOLD = 0.75


def fig_threshold_tradeoff() -> None:
    """Дефекты и ложные срабатывания от порога детектора."""
    thresholds = np.arange(0.40, 0.96, 0.05)
    esc_m, esc_h, fls_m, fls_h, duty = [], [], [], [], []
    for thr in thresholds:
        s = run_replications(replace(BASE, threshold=float(thr)), n_reps=N_REPS)
        esc_m.append(s.escaped_per_1000[0])
        esc_h.append(s.escaped_per_1000[1])
        fls_m.append(s.false_engagements[0])
        fls_h.append(s.false_engagements[1])
        duty.append(s.flap_duty[0])

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.errorbar(thresholds, esc_m, yerr=esc_h, marker="o", ms=4, capsize=3,
                 color="tab:red", label="Дефекты подачи, на 1000 деталей")
    ax1.set_xlabel("Порог уверенности детектора")
    ax1.set_ylabel("Дефекты подачи, на 1000 деталей", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax1.twinx()
    ax2.errorbar(thresholds, fls_m, yerr=fls_h, marker="s", ms=4, capsize=3,
                 color="tab:blue", label="Ложные срабатывания за прогон")
    ax2.set_ylabel("Ложные срабатывания заслонки (за 300 с)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax1.axvline(NIR_THRESHOLD, ls="--", color="0.3", lw=1.5)
    ax1.annotate("порог НИР 0,75", xy=(NIR_THRESHOLD, max(esc_m)),
                 xytext=(NIR_THRESHOLD + 0.01, max(esc_m) * 0.97), fontsize=9)
    ax1.set_title("Компромисс порога детектора (Monte Carlo, 95% ДИ)")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "flow_threshold_tradeoff.png", dpi=150)
    plt.close(fig)


def fig_capacity_curve() -> None:
    """Дефекты от интенсивности потока: с контролем (0,75) и без."""
    rates = np.arange(3.0, 8.1, 1.0)
    fig, ax = plt.subplots(figsize=(8, 5))
    for thr, label, color in (
        (NIR_THRESHOLD, "с контролем (порог 0,75)", "tab:green"),
        (0.9999, "без контроля", "tab:gray"),
    ):
        ys, hs = [], []
        for r in rates:
            s = run_replications(
                replace(BASE, arrival_rate_hz=float(r), threshold=thr), n_reps=N_REPS
            )
            ys.append(s.escaped_per_1000[0])
            hs.append(s.escaped_per_1000[1])
        ax.errorbar(rates, ys, yerr=hs, marker="o", ms=4, capsize=3, color=color, label=label)
    ax.axvline(5.0, ls="--", color="0.3", lw=1.5)
    ax.annotate("целевая производительность НИР: 5 дет/с", xy=(5.0, ax.get_ylim()[1] * 0.05),
                xytext=(5.1, ax.get_ylim()[1] * 0.05), fontsize=9)
    ax.set_xlabel("Интенсивность потока деталей, дет/с")
    ax.set_ylabel("Дефекты подачи, на 1000 деталей")
    ax.set_title("Дефекты подачи от интенсивности потока (Monte Carlo, 95% ДИ)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "flow_capacity_curve.png", dpi=150)
    plt.close(fig)


def fig_detector_quality() -> None:
    """Обратная задача: дефекты от качества детектора.

    Качество — средняя уверенность детектора на кадрах с нарушением:
    conf_true ~ Beta(a, 2), mean = a / (a + 2). Порог фиксирован 0,75.
    """
    a_values = np.array([3.0, 4.0, 6.0, 8.0, 12.0, 20.0])
    quality = a_values / (a_values + 2.0)
    ys, hs = [], []
    for a in a_values:
        s = run_replications(replace(BASE, conf_true_ab=(float(a), 2.0)), n_reps=N_REPS)
        ys.append(s.escaped_per_1000[0])
        hs.append(s.escaped_per_1000[1])
    # Уровень «без контроля» для сравнения
    s_blind = run_replications(replace(BASE, threshold=0.9999), n_reps=N_REPS)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(quality, ys, yerr=hs, marker="o", ms=4, capsize=3, color="tab:purple")
    ax.axhline(s_blind.escaped_per_1000[0], ls="--", color="0.4", lw=1.5,
               label="без контроля")
    ax.axvline(8.0 / 10.0, ls=":", color="0.3", lw=1.2)
    ax.annotate("базовое качество модели", xy=(0.8, max(ys) * 0.95), fontsize=9)
    ax.set_xlabel("Средняя уверенность детектора на нарушениях")
    ax.set_ylabel("Дефекты подачи, на 1000 деталей")
    ax.set_title("Требования к качеству детектора (порог 0,75; Monte Carlo, 95% ДИ)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "flow_detector_quality.png", dpi=150)
    plt.close(fig)


def print_summary() -> None:
    """Сводка по рабочей точке контура из НИР."""
    print("=" * 72)
    print("FeederTwin · Этап 2 · Контур контроля единичной подачи — сводка")
    print("=" * 72)
    s = run_replications(BASE, n_reps=N_REPS)
    s_blind = run_replications(replace(BASE, threshold=0.9999), n_reps=N_REPS)
    print(
        f"Рабочая точка (порог 0,75; 30 кадров/с; 2 кадра; привод 40 мс):\n"
        f"  производительность: {s.throughput_hz[0]:.2f} ± {s.throughput_hz[1]:.2f} дет/с\n"
        f"  дефекты подачи:     {s.escaped_per_1000[0]:.1f} ± {s.escaped_per_1000[1]:.1f}"
        f" на 1000 (без контроля: {s_blind.escaped_per_1000[0]:.1f})\n"
        f"  латентность коррекции: {s.mean_latency_s[0] * 1000:.0f} ±"
        f" {s.mean_latency_s[1] * 1000:.0f} мс (бюджет НИР: 186 мс)\n"
        f"  доля времени с заслонкой: {s.flap_duty[0] * 100:.1f}%\n"
        f"  ложные срабатывания: {s.false_engagements[0]:.0f} за 300 с"
    )
    print("=" * 72)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print_summary()
    print("Эксперименты Monte Carlo (займёт ~2–4 минуты)...")
    fig_threshold_tradeoff()
    print("  reports/figures/flow_threshold_tradeoff.png")
    fig_capacity_curve()
    print("  reports/figures/flow_capacity_curve.png")
    fig_detector_quality()
    print("  reports/figures/flow_detector_quality.png")
    print("Готово.")


if __name__ == "__main__":
    main()
