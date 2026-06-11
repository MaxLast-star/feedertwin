"""Тесты модуля вибротранспортирования."""

import math

import numpy as np
import pytest

from feedertwin.transport import TrayParams, gamma, mean_velocity, regime, simulate_particle, sweep

# Рабочая точка НИР: A = 0,275 мм, f = 16,7 Гц
WORK = TrayParams()
# Режим с подбрасыванием для проверки транспортирующих свойств
HOP = TrayParams(freq_hz=35.0, alpha_deg=8.0)


def test_gamma_working_point() -> None:
    """Γ = A·ω²/g для рабочей точки НИР: ручной расчёт даёт ≈ 0,3086."""
    omega = 2.0 * math.pi * 16.7
    expected = 0.275e-3 * omega**2 / 9.81
    assert gamma(WORK) == pytest.approx(expected, rel=1e-12)
    assert gamma(WORK) == pytest.approx(0.3086, rel=1e-3)


def test_regime_boundary() -> None:
    """Граница режимов проходит по Γ = 1."""
    below = TrayParams(freq_hz=30.0)   # Γ ≈ 0,996
    above = TrayParams(freq_hz=31.0)   # Γ ≈ 1,064
    assert gamma(below) < 1.0 < gamma(above)
    assert regime(below) == "no-hop"
    assert regime(above) == "hopping"


def test_no_transport_flat_tray_subhop() -> None:
    """Γ < 1 и α = 0: транспортирование отсутствует."""
    v = mean_velocity(TrayParams(alpha_deg=0.0), n_cycles=100)
    assert abs(v) < 1e-4


def test_no_transport_at_working_point() -> None:
    """Ключевой вывод модели: в рабочей точке НИР (Γ ≈ 0,31, tanα < μ)
    точечная кулоновская модель предсказывает нулевую скорость подачи."""
    v = mean_velocity(WORK, n_cycles=100)
    assert abs(v) < 1e-6


def test_velocity_monotone_in_alpha_hopping() -> None:
    """В режиме подбрасывания скорость не убывает с ростом угла наклона."""
    angles = np.arange(2.0, 13.0, 2.0)
    v = sweep(TrayParams(freq_hz=35.0), "alpha_deg", angles, n_cycles=100)
    assert np.all(np.diff(v) >= -1e-9)
    assert v[-1] > v[0] > 0.0


def test_velocity_grows_with_amplitude_hopping() -> None:
    """В режиме подбрасывания скорость растёт с амплитудой."""
    amps = [0.25e-3, 0.30e-3, 0.35e-3]
    v = sweep(TrayParams(freq_hz=35.0, alpha_deg=8.0), "amplitude_m", amps, n_cycles=100)
    assert v[0] < v[1] < v[2]


def test_particle_never_below_tray() -> None:
    """Деталь не проваливается под поверхность лотка."""
    trace = simulate_particle(HOP, n_cycles=50)
    assert float(trace.eta.min()) >= 0.0


def test_contact_flags_consistent() -> None:
    """В контакте нормальное отстояние равно нулю."""
    trace = simulate_particle(HOP, n_cycles=50)
    assert np.allclose(trace.eta[trace.contact], 0.0)
    # В режиме подбрасывания должны существовать фазы полёта
    assert not trace.contact.all()


def test_reproducibility() -> None:
    """Два прогона с одинаковыми параметрами идентичны (модель детерминирована)."""
    t1 = simulate_particle(HOP, n_cycles=30)
    t2 = simulate_particle(HOP, n_cycles=30)
    assert np.array_equal(t1.xi, t2.xi)
    assert np.array_equal(t1.xi_dot, t2.xi_dot)


def test_sweep_shape_and_finite() -> None:
    """Свип возвращает массив нужной длины без NaN/inf."""
    freqs = np.linspace(10.0, 40.0, 7)
    v = sweep(WORK, "freq_hz", freqs, n_cycles=60)
    assert v.shape == (7,)
    assert np.all(np.isfinite(v))


def test_invalid_inputs() -> None:
    """Некорректные аргументы дают понятные ошибки."""
    with pytest.raises(ValueError):
        sweep(WORK, "nonexistent_param", [1.0])
    with pytest.raises(ValueError):
        simulate_particle(WORK, n_cycles=0)


def test_steady_regime_detection() -> None:
    """Стационарность: покой и умеренное подбрасывание — устойчивы,
    соскальзывание при tanα > μ — разгонный (нестационарный) режим."""
    from feedertwin.transport import is_steady_regime

    assert is_steady_regime(WORK, n_cycles=100)                      # покой
    assert is_steady_regime(HOP, n_cycles=100)                       # hopping, периодический
    runaway = TrayParams(alpha_deg=18.0)                             # tan(18°) > 0,3
    assert not is_steady_regime(runaway, n_cycles=100)
