"""Аналитический модуль вибротранспортирования.

Численная модель установившегося движения одиночной детали по наклонному
лотку, совершающему вертикальные гармонические колебания (по схеме
двухвального дебалансного вибратора со встречным вращением валов:
горизонтальные составляющие возмущающей силы взаимно компенсируются).

Допущения и ограничения модели
------------------------------
1. Деталь — материальная точка (форма, ориентация и вращение не учитываются).
2. Трение детали о лоток — кулоновское с постоянным коэффициентом ``mu``
   (одинаковым для покоя и скольжения).
3. Удар при приземлении абсолютно неупругий по нормали (e = 0);
   тангенциальная составляющая скорости при ударе сохраняется.
4. Лоток — абсолютно жёсткая плоскость; упругие моды лотка, локальные
   деформации резиновой подкладки и микрорельеф (выступы) не моделируются.
5. Взаимодействие деталей между собой отсутствует (одиночная деталь);
   поток деталей и «сдвойки» моделируются в имитационном слое (этап 2).
6. Сопротивление воздуха пренебрежимо мало.

Кинематика
----------
Лоток наклонён под углом ``alpha`` к горизонту (подача под уклон) и
колеблется строго вертикально: s(t) = A·sin(ωt). Работа ведётся в
неинерциальной системе отсчёта, поступательно движущейся вместе с лотком;
оси: ξ — вдоль лотка (положительное направление — под уклон),
η — нормаль к лотку. Эффективное ускорение свободного падения:

    u(t) = g + s̈(t) = g − A·ω²·sin(ωt)

Контакт (η = 0, u > 0):
    нормальная реакция  N ∝ u·cos(alpha) ≥ 0
    скольжение:         ξ̈ = u·(sin(alpha) − mu·cos(alpha)·sign(ξ̇))
    залипание:          при ξ̇ = 0 и tan(alpha) ≤ mu деталь покоится
Отрыв: u(t) < 0, т. е. A·ω² > g (коэффициент режима Γ = A·ω²/g > 1).
Полёт: ξ̈ = u·sin(alpha), η̈ = −u·cos(alpha), до пересечения η = 0.

Важное следствие модели: для вертикальных колебаний условие отрыва
Γ = A·ω²/g не зависит от угла наклона (в отличие от классической схемы
с колебаниями, направленными под углом к лотку, где Γ = A·ω²/(g·cosα)).
При Γ ≤ 1 транспортирование возможно только при tan(alpha) > mu
(гравитационное соскальзывание); вибрация в этой модели сама по себе
не создаёт среднего движения, т. к. движущая сила и сила трения
пропорциональны одной и той же величине u(t).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace

import numpy as np

__all__ = [
    "TrayParams",
    "ParticleTrace",
    "gamma",
    "regime",
    "simulate_particle",
    "mean_velocity",
    "is_steady_regime",
    "sweep",
]

#: Порог «нулевой» скорости при численном залипании, м/с.
_STICK_EPS = 1e-9


@dataclass(frozen=True)
class TrayParams:
    """Параметры лотка и рабочей точки вибропривода.

    Значения по умолчанию — проектная рабочая точка из НИР:
    A = 0,275 мм (середина диапазона 0,25–0,3 мм), f = 16,7 Гц (1000 об/мин),
    угол наклона 6°, сталь/резиновая подкладка mu = 0,3.
    """

    amplitude_m: float = 0.275e-3
    freq_hz: float = 16.7
    alpha_deg: float = 6.0
    mu: float = 0.3
    g: float = 9.81

    @property
    def omega(self) -> float:
        """Круговая частота колебаний, рад/с."""
        return 2.0 * math.pi * self.freq_hz

    @property
    def alpha_rad(self) -> float:
        """Угол наклона лотка, рад."""
        return math.radians(self.alpha_deg)


@dataclass(frozen=True)
class ParticleTrace:
    """Результат прогона: траектория детали в осях лотка."""

    t: np.ndarray  #: время, с
    xi: np.ndarray  #: координата вдоль лотка (под уклон), м
    xi_dot: np.ndarray  #: скорость вдоль лотка, м/с
    eta: np.ndarray  #: нормальное отстояние от лотка, м
    contact: np.ndarray  #: bool, True — деталь в контакте с лотком


def gamma(params: TrayParams) -> float:
    """Коэффициент режима Γ = A·ω²/g.

    Для вертикальных колебаний наклонного лотка условие отрыва детали
    не содержит cos(alpha): отрыв наступает при Γ > 1.
    """
    return params.amplitude_m * params.omega**2 / params.g


def regime(params: TrayParams) -> str:
    """Качественный режим транспортирования: ``"no-hop"`` или ``"hopping"``."""
    return "hopping" if gamma(params) > 1.0 else "no-hop"


def simulate_particle(
    params: TrayParams,
    n_cycles: int = 200,
    steps_per_cycle: int = 2000,
) -> ParticleTrace:
    """Пошаговое интегрирование движения детали (явная схема, события).

    Args:
        params: параметры лотка и вибропривода.
        n_cycles: число периодов колебаний для прогона.
        steps_per_cycle: шагов интегрирования на период.

    Returns:
        ParticleTrace с массивами длины ``n_cycles * steps_per_cycle + 1``.
    """
    if n_cycles < 1 or steps_per_cycle < 100:
        raise ValueError("Требуется n_cycles >= 1 и steps_per_cycle >= 100")

    omega = params.omega
    sa, ca = math.sin(params.alpha_rad), math.cos(params.alpha_rad)
    mu, g, a = params.mu, params.g, params.amplitude_m
    period = 1.0 / params.freq_hz
    dt = period / steps_per_cycle
    n = n_cycles * steps_per_cycle

    t = np.linspace(0.0, n_cycles * period, n + 1)
    xi = np.zeros(n + 1)
    xi_dot = np.zeros(n + 1)
    eta = np.zeros(n + 1)
    contact = np.zeros(n + 1, dtype=bool)
    contact[0] = True

    x, v, h, hv = 0.0, 0.0, 0.0, 0.0  # ξ, ξ̇, η, η̇
    in_contact = True
    sticking = math.tan(params.alpha_rad) <= mu

    for i in range(1, n + 1):
        ti = t[i - 1]
        u = g - a * omega**2 * math.sin(omega * ti)

        if in_contact:
            if u < 0.0:
                # Отрыв: нормальная реакция обратилась в ноль.
                in_contact = False
                hv = 0.0
                ax = u * sa
                ah = -u * ca
                x += v * dt + 0.5 * ax * dt * dt
                v += ax * dt
                h += hv * dt + 0.5 * ah * dt * dt
                hv += ah * dt
            else:
                if abs(v) <= _STICK_EPS and sticking:
                    v = 0.0  # покой: трения покоя достаточно
                else:
                    sgn = 1.0 if v > _STICK_EPS else (-1.0 if v < -_STICK_EPS else 1.0)
                    acc = u * (sa - mu * ca * sgn)
                    v_new = v + acc * dt
                    # Прохождение через ноль при торможении трением → стоп
                    if sgn * v_new < 0.0 and sticking:
                        v_new = 0.0
                    x += 0.5 * (v + v_new) * dt
                    v = v_new
                h, hv = 0.0, 0.0
        else:
            ax = u * sa
            ah = -u * ca
            x += v * dt + 0.5 * ax * dt * dt
            v += ax * dt
            h_new = h + hv * dt + 0.5 * ah * dt * dt
            hv += ah * dt
            if h_new <= 0.0 and hv < 0.0:
                # Приземление: e = 0 по нормали, тангенциальная скорость сохраняется.
                h_new, hv = 0.0, 0.0
                in_contact = True
            h = max(h_new, 0.0)

        xi[i], xi_dot[i], eta[i], contact[i] = x, v, h, in_contact

    return ParticleTrace(t=t, xi=xi, xi_dot=xi_dot, eta=eta, contact=contact)


def mean_velocity(
    params: TrayParams,
    n_cycles: int = 200,
    steps_per_cycle: int = 2000,
) -> float:
    """Средняя установившаяся скорость детали вдоль лотка, м/с.

    Усреднение по последней половине прогона (переходный процесс отброшен).
    Положительное значение — движение под уклон (к зоне контроля).
    """
    trace = simulate_particle(params, n_cycles=n_cycles, steps_per_cycle=steps_per_cycle)
    half = len(trace.t) // 2
    dx = trace.xi[-1] - trace.xi[half]
    dt_total = trace.t[-1] - trace.t[half]
    return float(dx / dt_total)


def is_steady_regime(
    params: TrayParams,
    n_cycles: int = 200,
    steps_per_cycle: int = 2000,
    rel_tol: float = 0.10,
) -> bool:
    """Проверка существования установившегося режима транспортирования.

    Сравнивает средние скорости по третьей и четвёртой четвертям прогона.
    Если скорость продолжает расти (отличие более ``rel_tol``), режим
    нестационарный: деталь разгоняется (например, при tan(alpha) > mu —
    гравитационное соскальзывание, или при сильном подбрасывании на крутом
    лотке). В таком режиме «средняя скорость» зависит от длины лотка
    и не является характеристикой системы.
    """
    trace = simulate_particle(params, n_cycles=n_cycles, steps_per_cycle=steps_per_cycle)
    n = len(trace.t) - 1
    q2, q3 = n // 2, 3 * n // 4
    dt3 = trace.t[q3] - trace.t[q2]
    dt4 = trace.t[-1] - trace.t[q3]
    v3 = (trace.xi[q3] - trace.xi[q2]) / dt3
    v4 = (trace.xi[-1] - trace.xi[q3]) / dt4
    scale = max(abs(v3), abs(v4))
    if scale < 1e-7:  # обе скорости практически нулевые — стационарный покой
        return True
    return abs(v4 - v3) / scale <= rel_tol


def sweep(
    params: TrayParams,
    param_name: str,
    values: Iterable[float],
    n_cycles: int = 200,
    steps_per_cycle: int = 2000,
) -> np.ndarray:
    """Свип средней скорости по одному параметру ``TrayParams``.

    Args:
        params: базовый набор параметров.
        param_name: имя варьируемого поля (``amplitude_m``, ``freq_hz``,
            ``alpha_deg``, ``mu``).
        values: значения параметра.

    Returns:
        Массив средних скоростей той же длины, что ``values``.
    """
    if param_name not in TrayParams.__dataclass_fields__:
        raise ValueError(f"Неизвестный параметр: {param_name!r}")
    out = [
        mean_velocity(
            replace(params, **{param_name: float(val)}),
            n_cycles=n_cycles,
            steps_per_cycle=steps_per_cycle,
        )
        for val in values
    ]
    return np.asarray(out)
