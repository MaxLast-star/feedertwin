"""Имитационная модель потока деталей и контура контроля (этап 2).

Дискретно-событийная модель (SimPy) выходного участка вибрационного лотка
с замкнутым контуром контроля единичной подачи:

    поток деталей → зона контроля → камера/детектор → заслонка → коррекция

Временные параметры контура взяты из НИР:
- камера 30 кадров/с (период кадра ≈ 33,3 мс);
- решение по двум последовательным кадрам с уверенностью выше порога 0,75
  (t_подтв ≈ 66 мс);
- задержка привода заслонки 40 мс (серво SG90/MG90S, 20–40 мс);
- время стабилизации после ввода заслонки 50–120 мс;
- суммарный бюджет коррекции ≈ 186 мс ≈ 0,2 с;
- целевая производительность ≥ 5 дет/с.

Модель детектора
----------------
YOLOv8 в модели представлен стохастически: на каждом кадре детектор выдаёт
«уверенность» c ∈ (0, 1), что в зоне больше одной детали. Если нарушение
есть (в зоне ≥ 2 деталей), c ~ Beta(a_t, b_t) со средним вблизи 0,8;
если нарушения нет — c ~ Beta(a_f, b_f) со средним вблизи 0,2.
Срабатывание кадра: c > threshold. Это даёт настоящий компромисс
«пропуски ↔ ложные срабатывания», управляемый порогом, — как у реального
детектора. Параметры Beta — характеристики качества модели зрения,
варьируются в анализе чувствительности.

Допущения
---------
1. Интервалы между приходами деталей в зону — Gamma(k, θ): k задаёт
   регулярность потока (k = 1 — пуассоновский хаос, k → ∞ — конвейерная
   равномерность). Часть приходов — «сдвойки» (две детали с малым зазором).
2. Зона контроля — поле зрения камеры (~70 мм по НИР); деталь проходит
   его за transit_s = длина зоны / скорость подачи (по умолчанию 140 мс,
   что отвечает ~0,5 м/с). Скорость подачи — выход модели этапа 1.
3. Заслонка, введённая в зону, выталкивает лишние детали (все, кроме
   лидирующей) назад за линию заслонки с заданной вероятностью успеха
   и блокирует вход новых деталей до возврата; отсечённые детали входят
   повторно после возврата заслонки.
4. Дефект подачи («пропущенная сдвойка») — деталь, покинувшая зону раньше,
   чем через min_headway_s после предыдущей: нарушен минимальный
   технологический зазор поштучной выдачи.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import simpy

__all__ = [
    "FlowParams",
    "FlowMetrics",
    "ReplicationStats",
    "run_simulation",
    "run_replications",
]


@dataclass(frozen=True)
class FlowParams:
    """Параметры потока, детектора и исполнительного механизма.

    Значения по умолчанию — рабочая точка контура контроля из НИР.
    """

    # Поток деталей
    arrival_rate_hz: float = 5.0      #: средняя интенсивность прихода деталей, 1/с
    flow_shape_k: float = 4.0         #: параметр формы Gamma (регулярность потока)
    p_double: float = 0.08            #: вероятность, что приход — «сдвойка»
    double_gap_s: float = 0.015       #: зазор между деталями внутри сдвойки, с
    transit_s: float = 0.140          #: время прохождения зоны контроля, с
    min_headway_s: float = 0.060      #: мин. технологический зазор между выходами, с
    reentry_gap_s: float = 0.020      #: пауза перед повторным входом отсечённой детали, с
    entry_gap_s: float = 0.012        #: мин. физический зазор между входами в зону, с
                                      #  (длина детали / скорость подачи: ~6 мм / 0,5 м/с)

    # Детектор (модель машинного зрения)
    fps: float = 30.0                 #: частота кадров камеры, 1/с
    n_confirm_frames: int = 2         #: кадров подряд для подтверждения нарушения
    threshold: float = 0.75           #: порог уверенности
    conf_true_ab: tuple[float, float] = (8.0, 2.0)   #: Beta для кадров с нарушением
    conf_false_ab: tuple[float, float] = (2.0, 8.0)  #: Beta для кадров без нарушения

    # Исполнительный механизм
    actuation_delay_s: float = 0.040  #: задержка привода заслонки, с
    block_min_s: float = 0.050        #: мин. время удержания заслонки, с
    block_max_s: float = 0.120        #: макс. время удержания заслонки, с
    pushback_success: float = 0.95    #: вероятность успешного отсечения лишней детали

    # Прогон
    sim_time_s: float = 300.0         #: модельное время одного прогона, с
    seed: int = 0                     #: зерно генератора случайных чисел

    def validate(self) -> None:
        """Проверка корректности параметров."""
        if not 0.0 <= self.p_double <= 1.0:
            raise ValueError("p_double должен лежать в [0, 1]")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold должен лежать в (0, 1)")
        if self.arrival_rate_hz <= 0 or self.fps <= 0 or self.sim_time_s <= 0:
            raise ValueError("Интенсивности и время прогона должны быть положительными")
        if self.n_confirm_frames < 1:
            raise ValueError("n_confirm_frames >= 1")
        if self.min_headway_s >= self.transit_s:
            raise ValueError("min_headway_s должен быть меньше transit_s")


@dataclass
class FlowMetrics:
    """Сырые показатели одного прогона."""

    delivered: int = 0                #: деталей доставлено на выход
    doubles_generated: int = 0        #: сдвоек сгенерировано потоком
    escaped_doubles: int = 0          #: дефектов подачи (деталь вышла при занятой зоне)
    flap_engagements: int = 0         #: срабатываний заслонки всего
    false_engagements: int = 0        #: срабатываний без нарушения в зоне
    pushbacks: int = 0                #: деталей отсечено заслонкой
    total_block_time_s: float = 0.0   #: суммарное время удержания заслонки, с
    detection_latencies_s: list[float] = field(default_factory=list)
    #: задержки «возникновение нарушения → ввод заслонки», с
    control_trace: list[tuple[float, float]] = field(default_factory=list)
    #: трасса управления: (момент, множитель интенсивности потока)
    sim_time_s: float = 0.0

    @property
    def throughput_hz(self) -> float:
        """Фактическая производительность, дет/с."""
        return self.delivered / self.sim_time_s if self.sim_time_s > 0 else 0.0

    @property
    def escaped_per_1000(self) -> float:
        """Дефектов подачи на 1000 доставленных деталей."""
        return 1000.0 * self.escaped_doubles / self.delivered if self.delivered else 0.0

    @property
    def flap_duty(self) -> float:
        """Доля времени, в течение которого заслонка введена."""
        return self.total_block_time_s / self.sim_time_s if self.sim_time_s > 0 else 0.0

    @property
    def mean_detection_latency_s(self) -> float:
        """Средняя задержка коррекции, с (NaN, если срабатываний не было)."""
        if not self.detection_latencies_s:
            return float("nan")
        return float(np.mean(self.detection_latencies_s))


class _ZoneState:
    """Состояние зоны контроля и заслонки (разделяется процессами SimPy)."""

    def __init__(self, env: simpy.Environment) -> None:
        self.env = env
        self.parts: dict[int, simpy.Process] = {}     # id → процесс пребывания
        self.violation_since: float | None = None     # момент возникновения нарушения
        self.flap_engaged = False
        self.flap_released = env.event()
        self.flap_busy = False                        # от команды до возврата заслонки
        self.last_exit_t = -1e9                       # момент предыдущего выхода детали
        self.next_admit_t = -1e9                      # ранний момент следующего входа

    @property
    def count(self) -> int:
        return len(self.parts)

    def update_violation_clock(self) -> None:
        """Отметка момента возникновения/исчезновения нарушения (>1 детали)."""
        if self.count > 1 and self.violation_since is None:
            self.violation_since = self.env.now
        elif self.count <= 1:
            self.violation_since = None


def run_simulation(params: FlowParams, strategy=None) -> FlowMetrics:
    """Один прогон имитационной модели с заданным зерном.

    Args:
        params: параметры потока, детектора и механизма.
        strategy: стратегия управления (см. feedertwin.strategies).
            По умолчанию — BaselineThreshold(params.n_confirm_frames),
            в точности воспроизводящая поведение модели этапа 2.

    Returns:
        FlowMetrics с сырыми счётчиками и производными показателями.
    """
    from feedertwin.strategies import BaselineThreshold

    params.validate()
    if strategy is None:
        strategy = BaselineThreshold(n_confirm_frames=params.n_confirm_frames)
    strategy.reset()
    rng = np.random.default_rng(params.seed)
    env = simpy.Environment()
    zone = _ZoneState(env)
    m = FlowMetrics()
    next_id = iter(range(10**9))

    def part_dwell(pid: int) -> simpy.events.Event:
        """Пребывание детали в зоне: либо штатный выход, либо отсечение."""
        try:
            yield env.timeout(params.transit_s)
            # Штатный выход. Дефект — выход раньше min_headway после предыдущего.
            del zone.parts[pid]
            if env.now - zone.last_exit_t < params.min_headway_s:
                m.escaped_doubles += 1
            zone.last_exit_t = env.now
            m.delivered += 1
            zone.update_violation_clock()
        except simpy.Interrupt:
            # Отсечение заслонкой: деталь вытолкнута назад, вернётся после возврата.
            del zone.parts[pid]
            zone.update_violation_clock()
            m.pushbacks += 1
            env.process(reenter_after_release())

    def reenter_after_release() -> simpy.events.Event:
        """Повторный вход отсечённой детали после возврата заслонки."""
        if zone.flap_engaged:
            yield zone.flap_released
        yield env.timeout(params.reentry_gap_s + rng.uniform(0.0, params.reentry_gap_s))
        yield from admit_part()

    def admit_part() -> simpy.events.Event:
        """Вход детали в зону контроля.

        Ждёт возврата заслонки и соблюдает минимальный физический зазор
        между последовательными входами (детали не проходят сквозь
        друг друга): после снятия блокировки скопившиеся детали входят
        с интервалом entry_gap_s, а не одновременно. Стратегия может
        дополнительно дозировать вход через next_allowed.
        """
        while True:
            if zone.flap_engaged:
                yield zone.flap_released
                continue
            wait = max(zone.next_admit_t, strategy.next_allowed) - env.now
            if wait > 0:
                yield env.timeout(wait)
                continue  # после ожидания перепроверяем заслонку
            break
        zone.next_admit_t = env.now + params.entry_gap_s
        strategy.on_admission(env.now, rng)
        pid = next(next_id)
        zone.parts[pid] = env.process(part_dwell(pid))
        zone.update_violation_clock()

    def source() -> simpy.events.Event:
        """Источник деталей: Gamma-интервалы, часть приходов — сдвойки.

        Интенсивность масштабируется множителем стратегии (имитация
        регулирования интенсивности вибрации); изменения множителя
        записываются в трассу управления.
        """
        k = params.flow_shape_k
        last_factor = -1.0
        while True:
            factor = strategy.rate_factor(env.now)
            if factor != last_factor:
                m.control_trace.append((env.now, factor))
                last_factor = factor
            scale = 1.0 / (k * params.arrival_rate_hz * factor)
            yield env.timeout(rng.gamma(k, scale))
            yield from admit_part()
            if rng.random() < params.p_double:
                m.doubles_generated += 1
                yield env.timeout(params.double_gap_s)
                yield from admit_part()

    def flap_cycle(violation_started_at: float | None) -> simpy.events.Event:
        """Цикл заслонки: задержка привода → ввод → отсечение → возврат."""
        zone.flap_busy = True
        yield env.timeout(params.actuation_delay_s)
        engage_t = env.now
        if violation_started_at is not None:
            m.detection_latencies_s.append(engage_t - violation_started_at)
        m.flap_engagements += 1
        strategy.on_engagement(engage_t)
        if zone.count <= 1:
            m.false_engagements += 1
        zone.flap_engaged = True
        # Отсечение лишних деталей (все, кроме самой ранней)
        extras = sorted(zone.parts.keys())[1:]
        for pid in extras:
            if rng.random() < params.pushback_success:
                zone.parts[pid].interrupt()
        hold = rng.uniform(params.block_min_s, params.block_max_s)
        yield env.timeout(hold)
        m.total_block_time_s += hold
        zone.flap_engaged = False
        zone.flap_busy = False
        ev, zone.flap_released = zone.flap_released, env.event()
        ev.succeed()

    def camera() -> simpy.events.Event:
        """Камера + детектор: кадры с периодом 1/fps; решение — за стратегией."""
        a_t, b_t = params.conf_true_ab
        a_f, b_f = params.conf_false_ab
        frame_dt = 1.0 / params.fps
        while True:
            yield env.timeout(frame_dt)
            violation = zone.count > 1
            conf = rng.beta(a_t, b_t) if violation else rng.beta(a_f, b_f)
            positive = conf > params.threshold
            if strategy.on_frame(env.now, positive) and not zone.flap_busy:
                env.process(flap_cycle(zone.violation_since))

    env.process(source())
    env.process(camera())
    env.run(until=params.sim_time_s)
    m.sim_time_s = params.sim_time_s
    return m


@dataclass(frozen=True)
class ReplicationStats:
    """Статистика по серии независимых прогонов (Monte Carlo)."""

    n: int
    throughput_hz: tuple[float, float]        #: (среднее, полуширина 95% ДИ)
    escaped_per_1000: tuple[float, float]
    false_engagements: tuple[float, float]
    flap_duty: tuple[float, float]
    mean_latency_s: tuple[float, float]

    @staticmethod
    def _mean_ci(x: np.ndarray) -> tuple[float, float]:
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return float("nan"), float("nan")
        if len(x) == 1:
            return float(x[0]), float("nan")
        half = 1.96 * float(np.std(x, ddof=1)) / math.sqrt(len(x))
        return float(np.mean(x)), half


def run_replications(params: FlowParams, n_reps: int = 20, strategy=None) -> ReplicationStats:
    """Серия независимых прогонов с разными зёрнами и 95% ДИ по средним.

    Состояние стратегии сбрасывается моделью перед каждым прогоном.
    """
    if n_reps < 1:
        raise ValueError("n_reps >= 1")
    runs = [
        run_simulation(replace(params, seed=params.seed + i), strategy=strategy)
        for i in range(n_reps)
    ]
    g = ReplicationStats._mean_ci
    return ReplicationStats(
        n=n_reps,
        throughput_hz=g(np.array([r.throughput_hz for r in runs])),
        escaped_per_1000=g(np.array([r.escaped_per_1000 for r in runs])),
        false_engagements=g(np.array([float(r.false_engagements) for r in runs])),
        flap_duty=g(np.array([r.flap_duty for r in runs])),
        mean_latency_s=g(np.array([r.mean_detection_latency_s for r in runs])),
    )
