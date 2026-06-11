"""Стратегии управления контуром единичной подачи (этап 3).

Стратегия — подключаемый объект, который имитационная модель (flowsim)
опрашивает в трёх точках:

1. ``on_frame(now, positive)`` — на каждом кадре детектора; возвращает
   True, если нужно ввести заслонку;
2. ``rate_factor(now)`` — множитель интенсивности потока (0..1]; модель
   интерпретирует его как снижение интенсивности вибрации привода
   (упрощение: уменьшается частота прихода деталей в зону);
3. ``on_admission(now, rng)`` / ``next_allowed`` — дозирование входа:
   стратегия может задать ранний допустимый момент следующего входа.

Реализованы три стратегии:

- :class:`BaselineThreshold` — пороговая из НИР: N последовательных
  кадров с уверенностью выше порога → заслонка. Воспроизводит поведение
  модели этапа 2 бит-в-бит.
- :class:`AdaptiveVibration` — пороговая + обратная связь по частоте
  срабатываний: при перегрузке контура интенсивность вибрации снижается,
  в спокойном потоке — восстанавливается. Замкнутый контур
  «частота ошибок → интенсивность подачи».
- :class:`MeteringGate` — дозирующий выпуск: заслонка работает как шлюз,
  выдерживающий минимальный зазор между входами в зону (с заданной
  надёжностью посчёта деталей); детекторная защита сохраняется.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "BaselineThreshold",
    "AdaptiveVibration",
    "MeteringGate",
]


@dataclass
class BaselineThreshold:
    """Пороговая стратегия НИР: N кадров подряд выше порога → заслонка."""

    n_confirm_frames: int = 2

    _consecutive: int = field(default=0, repr=False)
    #: ранний допустимый момент следующего входа в зону (для дозирования)
    next_allowed: float = field(default=-1e9, repr=False)

    def reset(self) -> None:
        """Сброс состояния перед прогоном (вызывается моделью)."""
        self._consecutive = 0
        self.next_allowed = -1e9

    def on_frame(self, now: float, positive: bool) -> bool:
        """Кадр детектора: подтверждение N последовательными кадрами."""
        if positive:
            self._consecutive += 1
        else:
            self._consecutive = 0
        if self._consecutive >= self.n_confirm_frames:
            self._consecutive = 0
            return True
        return False

    def rate_factor(self, now: float) -> float:
        """Множитель интенсивности потока (базовая стратегия не вмешивается)."""
        return 1.0

    def on_admission(self, now: float, rng: np.random.Generator) -> None:
        """Вход детали в зону (базовая стратегия не дозирует)."""

    def on_engagement(self, now: float) -> None:
        """Уведомление о вводе заслонки (базовой стратегии не нужно)."""


@dataclass
class AdaptiveVibration(BaselineThreshold):
    """Пороговая стратегия + адаптация интенсивности вибрации.

    Обратная связь: скользящее окно срабатываний заслонки. Если за окно
    ``window_s`` произошло ≥ ``high_engagements`` срабатываний — интенсивность
    снижается умножением на ``down_step`` (не ниже ``min_factor``). Если
    срабатываний не было дольше ``quiet_s`` — восстанавливается умножением
    на ``up_step`` (не выше 1). Реализует принцип, предложенный в НИР
    (п. 5.4): «при частых ошибках снижать интенсивность вибрации».
    """

    window_s: float = 2.0          #: ширина окна подсчёта срабатываний, с
    high_engagements: int = 3      #: порог срабатываний за окно для снижения
    quiet_s: float = 3.0           #: тишина для шага восстановления, с
    down_step: float = 0.8         #: множитель снижения
    up_step: float = 1.05          #: множитель восстановления
    min_factor: float = 0.5        #: нижняя граница интенсивности

    _factor: float = field(default=1.0, repr=False)
    _engagements: list[float] = field(default_factory=list, repr=False)
    _last_change_t: float = field(default=0.0, repr=False)

    def reset(self) -> None:
        super().reset()
        self._factor = 1.0
        self._engagements = []
        self._last_change_t = 0.0

    def on_engagement(self, now: float) -> None:
        """Срабатывание заслонки: возможное снижение интенсивности."""
        self._engagements.append(now)
        cutoff = now - self.window_s
        self._engagements = [t for t in self._engagements if t >= cutoff]
        if len(self._engagements) >= self.high_engagements:
            self._factor = max(self.min_factor, self._factor * self.down_step)
            self._engagements = []
            self._last_change_t = now

    def rate_factor(self, now: float) -> float:
        """Текущий множитель с восстановлением в спокойном потоке."""
        last_engage = self._engagements[-1] if self._engagements else self._last_change_t
        if (
            self._factor < 1.0
            and now - last_engage >= self.quiet_s
            and now - self._last_change_t >= self.quiet_s
        ):
            self._factor = min(1.0, self._factor * self.up_step)
            self._last_change_t = now
        return self._factor


@dataclass
class MeteringGate(BaselineThreshold):
    """Дозирующий выпуск: шлюз выдерживает минимальный зазор между входами.

    После каждого зарегистрированного входа следующий вход разрешается
    не раньше, чем через ``meter_headway_s``. Регистрация входа происходит
    с надёжностью ``sense_reliability`` (пропуск посчёта — деталь проходит
    без выдержки). Поскольку время прохождения зоны одинаково для всех
    деталей, выдержанный зазор на входе сохраняется на выходе — дефекты
    возможны только при пропусках посчёта. Детекторная защита
    (пороговая логика) остаётся активной как вторая линия обороны.
    """

    meter_headway_s: float = 0.065   #: выдерживаемый зазор между входами, с
                                     #  (технологический 60 мс + запас 5 мс)
    sense_reliability: float = 0.98  #: вероятность зарегистрировать вход

    def on_admission(self, now: float, rng: np.random.Generator) -> None:
        """Регистрация входа: назначение раннего момента следующего входа."""
        if rng.random() < self.sense_reliability:
            self.next_allowed = now + self.meter_headway_s
