"""Сценарий исследования модели вибротранспортирования.

Запуск:  python scripts/run_transport_study.py

Строит три графика в reports/figures/ и печатает сводку по рабочей
точке НИР (A = 0,25–0,3 мм, f = 16,7 Гц, α = 0–15°).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from feedertwin.transport import (  # noqa: E402
    TrayParams,
    gamma,
    is_steady_regime,
    mean_velocity,
    regime,
    sweep,
)

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
WORK_FREQ = 16.7  # Гц, 1000 об/мин — рабочая точка НИР
WORK_AMPS_MM = (0.25, 0.275, 0.30)  # мм — диапазон амплитуд из НИР


def masked_sweep(base: TrayParams, name: str, values: np.ndarray) -> np.ndarray:
    """Свип скорости с маскировкой нестационарных (разгонных) режимов NaN."""
    from dataclasses import replace

    v = sweep(base, name, values) * 1000.0
    steady = np.array(
        [is_steady_regime(replace(base, **{name: float(x)}), n_cycles=120) for x in values]
    )
    return np.where(steady, v, np.nan)


def fig_velocity_vs_alpha() -> None:
    """v(α) при разных частотах: рабочая точка против режима подбрасывания."""
    angles = np.arange(0.0, 15.0, 1.0)
    fig, ax = plt.subplots(figsize=(8, 5))
    for f_hz in (WORK_FREQ, 31.0, 35.0):
        p = TrayParams(freq_hz=f_hz)
        v_plot = masked_sweep(p, "alpha_deg", angles)
        label = f"f = {f_hz:g} Гц (Γ = {gamma(p):.2f}, {regime(p)})"
        lw = 2.5 if f_hz == WORK_FREQ else 1.8
        ax.plot(angles, v_plot, marker="o", ms=3, lw=lw, label=label)
        if np.isnan(v_plot).any():
            a_run = float(angles[np.isnan(v_plot)].min())
            ax.axvline(a_run, ls="--", color="0.4", lw=1)
            ax.annotate("зона разгона\n(нет устан. режима)", xy=(a_run + 0.15, 0.0),
                        fontsize=8, color="0.35")
    ax.set_xlabel("Угол наклона лотка α, °")
    ax.set_ylabel("Средняя скорость подачи, мм/с")
    ax.set_title("Скорость подачи от угла наклона (A = 0,275 мм)")
    ax.axvspan(0, 15, color="0.93", zorder=0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "velocity_vs_alpha.png", dpi=150)
    plt.close(fig)


def fig_velocity_vs_freq() -> None:
    """v(f) для трёх амплитуд из НИР: виден порог Γ = 1."""
    freqs = np.arange(10.0, 45.1, 1.0)
    fig, ax = plt.subplots(figsize=(8, 5))
    for a_mm in WORK_AMPS_MM:
        p = TrayParams(amplitude_m=a_mm * 1e-3, alpha_deg=10.0)
        v = masked_sweep(p, "freq_hz", freqs)
        ax.plot(freqs, v, marker="o", ms=3, label=f"A = {a_mm:g} мм")
        # Частота порога подбрасывания Γ=1 для данной амплитуды
        f_crit = float(np.sqrt(9.81 / (a_mm * 1e-3)) / (2 * np.pi))
        ax.axvline(f_crit, ls=":", color="0.5", lw=1)
    ax.axvline(WORK_FREQ, ls="--", color="crimson", lw=1.5,
               label=f"Рабочая точка НИР: {WORK_FREQ} Гц")
    ax.set_xlabel("Частота колебаний f, Гц")
    ax.set_ylabel("Средняя скорость подачи, мм/с")
    ax.set_title("Скорость подачи от частоты (α = 10°); пунктир — порог Γ = 1;\n"
                 "разрывы кривых — нестационарные (разгонные) режимы")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "velocity_vs_freq.png", dpi=150)
    plt.close(fig)


def fig_sensitivity_mu() -> None:
    """Чувствительность к коэффициенту трения в режиме подбрасывания."""
    mus = np.arange(0.10, 0.61, 0.05)
    fig, ax = plt.subplots(figsize=(8, 5))
    for f_hz, a_deg in ((35.0, 8.0), (35.0, 12.0), (40.0, 8.0)):
        p = TrayParams(freq_hz=f_hz, alpha_deg=a_deg)
        v = masked_sweep(p, "mu", mus)
        ax.plot(mus, v, marker="o", ms=3, label=f"f = {f_hz:g} Гц, α = {a_deg:g}°")
    ax.set_xlabel("Коэффициент трения μ")
    ax.set_ylabel("Средняя скорость подачи, мм/с")
    ax.set_title("Чувствительность скорости к трению (режим подбрасывания)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sensitivity_mu.png", dpi=150)
    plt.close(fig)


def print_summary() -> None:
    """Сводка по рабочей точке НИР."""
    print("=" * 72)
    print("FeederTwin · Этап 1 · Модель вибротранспортирования — сводка")
    print("=" * 72)
    for a_mm in WORK_AMPS_MM:
        p = TrayParams(amplitude_m=a_mm * 1e-3, freq_hz=WORK_FREQ)
        print(f"A = {a_mm:g} мм, f = {WORK_FREQ} Гц → Γ = {gamma(p):.3f} ({regime(p)})")
    p = TrayParams()
    v6 = mean_velocity(p) * 1000.0
    print(f"\nСредняя скорость в рабочей точке (α = 6°): {v6:.4f} мм/с")
    f_crit = float(np.sqrt(9.81 / p.amplitude_m) / (2 * np.pi))
    a_crit = 9.81 / p.omega**2 * 1000.0
    print(
        "\nВывод: рабочая точка находится в безотрывном режиме (Γ < 1).\n"
        "Точечная кулоновская модель в этом режиме не даёт среднего движения\n"
        "при tg(α) ≤ μ: движущая сила и трение пропорциональны одной величине\n"
        "u(t) = g + s̈(t). Порог режима подбрасывания (Γ = 1) достигается при\n"
        f"f ≈ {f_crit:.1f} Гц (при A = 0,275 мм) либо A ≈ {a_crit:.2f} мм (при 16,7 Гц).\n"
        "Гипотезы о механизме подачи в реальной системе и рекомендации —\n"
        "см. docs/ (отчёт этапа 1)."
    )
    print("=" * 72)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print_summary()
    print("Построение графиков (займёт ~1–2 минуты)...")
    fig_velocity_vs_alpha()
    print("  reports/figures/velocity_vs_alpha.png")
    fig_velocity_vs_freq()
    print("  reports/figures/velocity_vs_freq.png")
    fig_sensitivity_mu()
    print("  reports/figures/sensitivity_mu.png")
    print("Готово.")


if __name__ == "__main__":
    main()
