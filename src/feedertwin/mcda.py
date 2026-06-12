"""Многокритериальный анализ решений: AHP и TOPSIS (этап 4).

Реализованы «с нуля» (без сторонних MCDA-библиотек, для прозрачности
вычислений):

- :func:`ahp_weights` — метод анализа иерархий (Saaty): веса критериев
  как главный собственный вектор матрицы парных сравнений, с расчётом
  индекса (CI) и отношения (CR) согласованности;
- :func:`topsis` — ранжирование альтернатив по близости к идеальному
  решению (векторная нормализация, евклидовы расстояния);
- :func:`topsis_weight_sweep` — анализ чувствительности: свип веса одного
  критерия (остальные масштабируются пропорционально) для поиска точек
  смены ранга.

Соглашения: матрица решений X имеет форму (альтернативы × критерии);
``benefit[j]`` — True, если критерий j максимизируется, False — если
минимизируется (стоимостной).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "AhpResult",
    "TopsisResult",
    "ahp_weights",
    "topsis",
    "topsis_weight_sweep",
]

#: Случайные индексы согласованности Саати (RI) для n = 1..10.
_SAATY_RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12,
             6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

#: Принятый порог отношения согласованности.
CR_THRESHOLD = 0.10


@dataclass(frozen=True)
class AhpResult:
    """Результат AHP: веса, собственное число и показатели согласованности."""

    weights: np.ndarray      #: нормированные веса критериев (сумма = 1)
    lambda_max: float        #: главное собственное число матрицы
    ci: float                #: индекс согласованности (λ_max − n)/(n − 1)
    cr: float                #: отношение согласованности CI/RI

    @property
    def is_consistent(self) -> bool:
        """Согласована ли матрица сравнений (CR ≤ 0,10)."""
        return self.cr <= CR_THRESHOLD


def _validate_pairwise(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("Матрица парных сравнений должна быть квадратной")
    n = m.shape[0]
    if n < 2 or n > 10:
        raise ValueError("Поддерживаются матрицы размера 2..10 (таблица RI Саати)")
    if np.any(m <= 0):
        raise ValueError("Все элементы матрицы должны быть положительными")
    if not np.allclose(np.diag(m), 1.0):
        raise ValueError("Диагональ матрицы парных сравнений должна состоять из единиц")
    if not np.allclose(m * m.T, 1.0, rtol=1e-6):
        raise ValueError("Матрица должна быть обратносимметричной: a_ij = 1/a_ji")
    return m


def ahp_weights(matrix: np.ndarray) -> AhpResult:
    """Веса критериев методом анализа иерархий (Saaty).

    Args:
        matrix: квадратная обратносимметричная матрица парных сравнений
            (шкала Саати 1..9; a_ij — во сколько раз критерий i важнее j).

    Returns:
        AhpResult с весами (главный собственный вектор, нормированный
        на единичную сумму) и показателями согласованности.
    """
    m = _validate_pairwise(matrix)
    n = m.shape[0]
    eigvals, eigvecs = np.linalg.eig(m)
    k = int(np.argmax(eigvals.real))
    lambda_max = float(eigvals[k].real)
    w = np.abs(eigvecs[:, k].real)
    w = w / w.sum()
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri = _SAATY_RI[n]
    cr = ci / ri if ri > 0 else 0.0
    return AhpResult(weights=w, lambda_max=lambda_max, ci=float(ci), cr=float(cr))


@dataclass(frozen=True)
class TopsisResult:
    """Результат TOPSIS."""

    closeness: np.ndarray    #: близость к идеальному решению, 0..1
    ranking: np.ndarray      #: индексы альтернатив от лучшей к худшей
    d_plus: np.ndarray       #: расстояния до идеального решения
    d_minus: np.ndarray      #: расстояния до антиидеального решения


def topsis(
    decision_matrix: np.ndarray,
    weights: np.ndarray,
    benefit: list[bool] | np.ndarray,
) -> TopsisResult:
    """Ранжирование альтернатив методом TOPSIS.

    Args:
        decision_matrix: матрица (альтернативы × критерии), сырые значения.
        weights: веса критериев (нормируются на единичную сумму).
        benefit: для каждого критерия True (максимизация) или False
            (минимизация).

    Returns:
        TopsisResult: близости C_i = d⁻/(d⁺ + d⁻) и ранжирование.
    """
    x = np.asarray(decision_matrix, dtype=float)
    w = np.asarray(weights, dtype=float)
    b = np.asarray(benefit, dtype=bool)
    if x.ndim != 2:
        raise ValueError("decision_matrix должна быть двумерной")
    n_alt, n_crit = x.shape
    if n_alt < 2:
        raise ValueError("Нужно не менее двух альтернатив")
    if w.shape != (n_crit,) or b.shape != (n_crit,):
        raise ValueError("Размеры weights и benefit должны равняться числу критериев")
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Веса должны быть неотрицательными с положительной суммой")
    w = w / w.sum()

    # Векторная нормализация; нулевой столбец оставляем нулевым
    norms = np.linalg.norm(x, axis=0)
    norms[norms == 0] = 1.0
    v = (x / norms) * w

    ideal = np.where(b, v.max(axis=0), v.min(axis=0))
    anti = np.where(b, v.min(axis=0), v.max(axis=0))
    d_plus = np.linalg.norm(v - ideal, axis=1)
    d_minus = np.linalg.norm(v - anti, axis=1)
    denom = d_plus + d_minus
    denom[denom == 0] = 1.0  # альтернатива, совпавшая с обоими полюсами
    closeness = d_minus / denom
    ranking = np.argsort(-closeness, kind="stable")
    return TopsisResult(closeness=closeness, ranking=ranking,
                        d_plus=d_plus, d_minus=d_minus)


def topsis_weight_sweep(
    decision_matrix: np.ndarray,
    weights: np.ndarray,
    benefit: list[bool] | np.ndarray,
    criterion_index: int,
    grid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Чувствительность TOPSIS к весу одного критерия.

    Вес критерия ``criterion_index`` пробегает сетку ``grid`` (по умолчанию
    0,02..0,98), остальные веса масштабируются пропорционально исходным.

    Returns:
        (grid, closeness_matrix) — closeness_matrix имеет форму
        (len(grid), n_alternatives).
    """
    w0 = np.asarray(weights, dtype=float)
    w0 = w0 / w0.sum()
    n_crit = len(w0)
    if not 0 <= criterion_index < n_crit:
        raise ValueError("criterion_index вне диапазона")
    if grid is None:
        grid = np.linspace(0.02, 0.98, 49)
    others = np.delete(w0, criterion_index)
    if others.sum() <= 0:
        raise ValueError("Остальные веса не могут быть все нулевыми")
    others = others / others.sum()

    rows = []
    for wc in grid:
        w = np.empty(n_crit)
        w[criterion_index] = wc
        w[np.arange(n_crit) != criterion_index] = (1.0 - wc) * others
        rows.append(topsis(decision_matrix, w, benefit).closeness)
    return np.asarray(grid), np.asarray(rows)
