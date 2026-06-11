"""Тесты стратегий управления."""

from dataclasses import replace

import pytest

from feedertwin.flowsim import FlowParams, run_simulation
from feedertwin.strategies import AdaptiveVibration, BaselineThreshold, MeteringGate

BASE = FlowParams(sim_time_s=120.0, seed=11)
HEAVY = replace(BASE, arrival_rate_hz=7.0, sim_time_s=200.0)


def test_baseline_strategy_is_backward_compatible() -> None:
    """Явная BaselineThreshold даёт бит-в-бит то же, что модель этапа 2."""
    m_default = run_simulation(BASE)
    m_explicit = run_simulation(BASE, strategy=BaselineThreshold(n_confirm_frames=2))
    assert (m_default.delivered, m_default.escaped_doubles, m_default.flap_engagements) == (
        m_explicit.delivered,
        m_explicit.escaped_doubles,
        m_explicit.flap_engagements,
    )


def test_strategy_state_resets_between_runs() -> None:
    """Повторный прогон с тем же объектом стратегии воспроизводим."""
    strat = AdaptiveVibration()
    m1 = run_simulation(HEAVY, strategy=strat)
    m2 = run_simulation(HEAVY, strategy=strat)
    assert m1.delivered == m2.delivered
    assert m1.escaped_doubles == m2.escaped_doubles


def test_metering_eliminates_defects_with_perfect_sensing() -> None:
    """Дозирование с идеальным посчётом: ноль дефектов подачи."""
    strat = MeteringGate(sense_reliability=1.0)
    m = run_simulation(replace(BASE, sim_time_s=300.0), strategy=strat)
    assert m.escaped_doubles == 0
    assert m.delivered > 0


def test_metering_with_misses_still_beats_baseline() -> None:
    """Дозирование с надёжностью 0,9 — дефектов меньше трети от базовой."""
    m_meter = run_simulation(BASE, strategy=MeteringGate(sense_reliability=0.9))
    m_base = run_simulation(BASE)
    assert 0 < m_meter.escaped_per_1000 < m_base.escaped_per_1000 / 3.0


def test_metering_keeps_throughput() -> None:
    """Дозирование не душит производительность в рабочей точке (±10%)."""
    m = run_simulation(replace(BASE, sim_time_s=300.0), strategy=MeteringGate())
    expected = BASE.arrival_rate_hz * (1.0 + BASE.p_double)
    assert m.throughput_hz == pytest.approx(expected, rel=0.10)


def test_adaptive_reduces_rate_under_overload() -> None:
    """При перегрузке адаптивная стратегия снижает интенсивность потока."""
    m = run_simulation(HEAVY, strategy=AdaptiveVibration())
    factors = [f for _, f in m.control_trace]
    assert min(factors) < 0.9
    assert m.throughput_hz < HEAVY.arrival_rate_hz  # часть производительности отдана


def test_adaptive_recovers_in_light_flow() -> None:
    """В лёгком потоке снижения редки и фактор восстанавливается до 1,0.

    Финальная точка трассы может попасть на середину восстановления после
    случайного кластера срабатываний, поэтому проверяются два устойчивых
    свойства: (1) после первого снижения фактор хотя бы раз возвращается
    к 1,0; (2) средневзвешенный по времени фактор близок к единице.
    """
    light = replace(BASE, arrival_rate_hz=2.0, sim_time_s=200.0)
    m = run_simulation(light, strategy=AdaptiveVibration())
    trace = m.control_trace
    drops = [i for i, (_, f) in enumerate(trace) if f < 0.999]
    if drops:  # снижение случилось — должно быть и восстановление
        assert any(f >= 0.999 for _, f in trace[drops[0] + 1 :])
    # Средневзвешенный по времени фактор
    total = 0.0
    for (t0, f), (t1, _) in zip(trace, trace[1:] + [(light.sim_time_s, 0.0)], strict=False):
        total += f * (t1 - t0)
    assert total / light.sim_time_s > 0.9


def test_adaptive_cuts_defects_under_overload() -> None:
    """Парное сравнение @7 дет/с: адаптивная стратегия снижает дефекты."""
    gains = []
    for seed in range(11, 17):
        p = replace(HEAVY, seed=seed)
        m_base = run_simulation(p)
        m_adapt = run_simulation(p, strategy=AdaptiveVibration())
        gains.append(m_base.escaped_per_1000 - m_adapt.escaped_per_1000)
    assert sum(gains) / len(gains) > 0.0


def test_control_trace_recorded() -> None:
    """Трасса управления непуста и начинается с фактора 1,0."""
    m = run_simulation(BASE, strategy=AdaptiveVibration())
    assert len(m.control_trace) >= 1
    assert m.control_trace[0][1] == pytest.approx(1.0)
