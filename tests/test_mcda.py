"""Тесты модуля многокритериального анализа (AHP, TOPSIS)."""

import numpy as np
import pytest

from feedertwin.mcda import ahp_weights, topsis, topsis_weight_sweep

# Идеально согласованная матрица 3×3: веса 4/7, 2/7, 1/7
CONSISTENT = np.array([[1.0, 2.0, 4.0], [0.5, 1.0, 2.0], [0.25, 0.5, 1.0]])


def test_ahp_exact_weights_for_consistent_matrix() -> None:
    """Для согласованной матрицы веса восстанавливаются точно, CR ≈ 0."""
    r = ahp_weights(CONSISTENT)
    assert np.allclose(r.weights, [4 / 7, 2 / 7, 1 / 7], atol=1e-9)
    assert r.lambda_max == pytest.approx(3.0, abs=1e-9)
    assert abs(r.cr) < 1e-9
    assert r.is_consistent


def test_ahp_saaty_reference_example() -> None:
    """Классический пример со шкалой Саати: λ ≈ 3,039, CR ≈ 0,033."""
    m = np.array([[1, 3, 5], [1 / 3, 1, 3], [1 / 5, 1 / 3, 1]])
    r = ahp_weights(m)
    assert r.lambda_max == pytest.approx(3.039, abs=0.005)
    assert r.cr == pytest.approx(0.033, abs=0.005)
    assert np.allclose(r.weights, [0.637, 0.258, 0.105], atol=0.005)


def test_ahp_detects_inconsistency() -> None:
    """Циклическая матрица (A>B>C>A) даёт CR выше порога 0,10."""
    m = np.array([[1, 5, 1 / 5], [1 / 5, 1, 5], [5, 1 / 5, 1]])
    r = ahp_weights(m)
    assert r.cr > 0.10
    assert not r.is_consistent


def test_ahp_weights_sum_to_one() -> None:
    """Веса нормированы на единичную сумму."""
    r = ahp_weights(CONSISTENT)
    assert r.weights.sum() == pytest.approx(1.0)
    assert np.all(r.weights > 0)


def test_ahp_validation() -> None:
    """Невалидные матрицы отклоняются с понятными ошибками."""
    with pytest.raises(ValueError):
        ahp_weights(np.array([[1.0, 2.0]]))                      # не квадратная
    with pytest.raises(ValueError):
        ahp_weights(np.array([[1.0, -2.0], [-0.5, 1.0]]))        # отрицательные
    with pytest.raises(ValueError):
        ahp_weights(np.array([[2.0, 1.0], [1.0, 1.0]]))          # диагональ ≠ 1
    with pytest.raises(ValueError):
        ahp_weights(np.array([[1.0, 3.0], [0.5, 1.0]]))          # не обратносимметрична


def test_topsis_dominant_alternative_gets_unity() -> None:
    """Альтернатива, лучшая по всем критериям, получает близость 1."""
    x = np.array([[10.0, 1.0], [5.0, 2.0]])
    t = topsis(x, [0.5, 0.5], [True, False])
    assert t.closeness[0] == pytest.approx(1.0)
    assert t.closeness[1] == pytest.approx(0.0)
    assert list(t.ranking) == [0, 1]


def test_topsis_symmetric_tradeoff() -> None:
    """Симметричный размен при равных весах даёт равные близости."""
    x = np.array([[10.0, 2.0], [5.0, 1.0]])
    t = topsis(x, [0.5, 0.5], [True, False])
    assert t.closeness[0] == pytest.approx(t.closeness[1])


def test_topsis_scale_invariance() -> None:
    """Ранжирование не зависит от единиц измерения критериев."""
    x = np.array([[100.0, 5.0, 3.0], [80.0, 7.0, 4.0], [60.0, 2.0, 5.0]])
    w = [0.5, 0.3, 0.2]
    b = [True, False, True]
    t1 = topsis(x, w, b)
    scaled = x * np.array([0.001, 1000.0, 7.5])  # смена единиц по столбцам
    t2 = topsis(scaled, w, b)
    assert list(t1.ranking) == list(t2.ranking)
    assert np.allclose(t1.closeness, t2.closeness)


def test_topsis_weight_shifts_ranking() -> None:
    """Сдвиг веса в пользу критерия меняет лидера предсказуемо."""
    x = np.array([[10.0, 10.0], [1.0, 1.0]])  # A лучше по 1-му (benefit), хуже по 2-му (cost)
    only_first = topsis(x, [0.99, 0.01], [True, False])
    only_second = topsis(x, [0.01, 0.99], [True, False])
    assert only_first.ranking[0] == 0
    assert only_second.ranking[0] == 1


def test_topsis_validation() -> None:
    """Невалидные входы отклоняются."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError):
        topsis(np.array([[1.0, 2.0]]), [0.5, 0.5], [True, True])  # одна альтернатива
    with pytest.raises(ValueError):
        topsis(x, [0.5], [True, True])                            # размер весов
    with pytest.raises(ValueError):
        topsis(x, [-0.5, 1.5], [True, True])                      # отрицательный вес


def test_weight_sweep_shape_and_consistency() -> None:
    """Свип веса: правильная форма и согласие с точечным расчётом."""
    x = np.array([[10.0, 2.0, 3.0], [5.0, 1.0, 6.0], [7.0, 3.0, 1.0]])
    w = np.array([0.4, 0.3, 0.3])
    b = [True, False, True]
    grid, closeness = topsis_weight_sweep(x, w, b, criterion_index=0)
    assert closeness.shape == (len(grid), 3)
    # Точка сетки, совпадающая с исходным весом, должна дать тот же результат
    i = int(np.argmin(np.abs(grid - 0.4)))
    w_check = np.array([grid[i], 0.5 * (1 - grid[i]), 0.5 * (1 - grid[i])])
    assert np.allclose(closeness[i], topsis(x, w_check, b).closeness)
