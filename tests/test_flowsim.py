"""Тесты имитационной модели потока и контура контроля."""

from dataclasses import replace

import numpy as np
import pytest

from feedertwin.flowsim import FlowParams, run_replications, run_simulation

# Базовая точка НИР; короткие прогоны для скорости тестов
BASE = FlowParams(sim_time_s=120.0, seed=7)
# «Слепой» детектор: порог недостижим для Beta-уверенности
BLIND = replace(BASE, threshold=0.9999)
# Почти идеальный детектор: уверенности разнесены, порог посередине
SHARP = replace(BASE, conf_true_ab=(50.0, 1.0), conf_false_ab=(1.0, 50.0), threshold=0.5)


def test_validation_errors() -> None:
    """Некорректные параметры дают понятные ошибки."""
    with pytest.raises(ValueError):
        run_simulation(replace(BASE, p_double=1.5))
    with pytest.raises(ValueError):
        run_simulation(replace(BASE, threshold=1.2))
    with pytest.raises(ValueError):
        run_simulation(replace(BASE, min_headway_s=0.5, transit_s=0.1))
    with pytest.raises(ValueError):
        run_replications(BASE, n_reps=0)


def test_reproducibility() -> None:
    """Одинаковое зерно → идентичные результаты."""
    m1 = run_simulation(BASE)
    m2 = run_simulation(BASE)
    assert (m1.delivered, m1.escaped_doubles, m1.flap_engagements, m1.pushbacks) == (
        m2.delivered,
        m2.escaped_doubles,
        m2.flap_engagements,
        m2.pushbacks,
    )


def test_regular_clean_flow_has_no_defects() -> None:
    """Регулярный поток без сдвоек и без срабатываний — ноль дефектов."""
    p = replace(BLIND, p_double=0.0, flow_shape_k=200.0)
    m = run_simulation(p)
    assert m.escaped_doubles == 0
    assert m.flap_engagements == 0
    assert m.delivered > 0


def test_blind_detector_passes_doubles() -> None:
    """Без контроля большинство сдвоек становится дефектами подачи."""
    m = run_simulation(BLIND)
    assert m.doubles_generated > 0
    assert m.escaped_doubles >= 0.8 * m.doubles_generated
    assert m.pushbacks == 0


def test_control_reduces_defects() -> None:
    """Контур контроля (порог 0,75 из НИР) снижает дефекты против «слепого».

    Парное сравнение: одинаковые зёрна → одинаковые реализации потока,
    различие создаёт только контур управления.
    """
    diffs = []
    for seed in range(7, 15):
        m_ctrl = run_simulation(replace(BASE, seed=seed))
        m_blind = run_simulation(replace(BLIND, seed=seed))
        diffs.append(m_blind.escaped_per_1000 - m_ctrl.escaped_per_1000)
    mean_gain = float(np.mean(diffs))
    assert mean_gain > 0.0
    # Снижение хотя бы на 5% от уровня «слепого» детектора в среднем
    blind_level = np.mean(
        [run_simulation(replace(BLIND, seed=s)).escaped_per_1000 for s in range(7, 15)]
    )
    assert mean_gain >= 0.05 * blind_level


def test_sharp_detector_better_than_baseline() -> None:
    """Более качественный детектор → меньше дефектов, чем у базового."""
    s_sharp = run_replications(SHARP, n_reps=6)
    s_base = run_replications(BASE, n_reps=6)
    assert s_sharp.escaped_per_1000[0] < s_base.escaped_per_1000[0]


def test_detection_latency_matches_budget() -> None:
    """Задержка «нарушение → ввод заслонки» отвечает бюджету НИР:
    ~2 кадра подтверждения (66 мс) + привод (40 мс) ≈ 0,07–0,15 с."""
    m = run_simulation(replace(SHARP, sim_time_s=300.0))
    assert m.flap_engagements > 10
    assert 0.07 <= m.mean_detection_latency_s <= 0.15


def test_threshold_controls_false_engagements() -> None:
    """Рост порога снижает число ложных срабатываний."""
    lo = run_replications(replace(BASE, threshold=0.5), n_reps=5)
    hi = run_replications(replace(BASE, threshold=0.95), n_reps=5)
    assert hi.false_engagements[0] < lo.false_engagements[0]


def test_throughput_matches_arrival_rate() -> None:
    """Производительность ≈ интенсивность прихода с учётом сдвоек (±10%)."""
    m = run_simulation(replace(BASE, sim_time_s=300.0))
    expected = BASE.arrival_rate_hz * (1.0 + BASE.p_double)
    assert m.throughput_hz == pytest.approx(expected, rel=0.10)


def test_metric_bounds() -> None:
    """Производные метрики в допустимых диапазонах."""
    m = run_simulation(BASE)
    assert 0.0 <= m.flap_duty <= 1.0
    assert m.delivered > 0
    assert m.escaped_doubles <= m.delivered
    s = run_replications(BASE, n_reps=3)
    assert np.isfinite(s.throughput_hz[0])
