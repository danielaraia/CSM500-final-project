import numpy as np
import pytest

from graphs.mi import (
    MI_NEIGHBORS,
    MI_RANDOM_STATE,
    build_mi_graph,
    mutual_information,
)
from graphs.pearson import (
    ELECTRICAL_SAMPLES_1S,
    ELECTRICAL_SAMPLES_4S,
    MOVEMENT_SAMPLES_4S,
    N_ELECTRICAL_NODES,
    N_MOVEMENT_NODES,
    N_NODES,
    series_for_pair,
)


def test_estimator_settings_match_the_lock():
    assert MI_NEIGHBORS == 3
    assert MI_RANDOM_STATE == 123


def test_independent_noise_is_near_zero():
    rng = np.random.default_rng(123)
    series_a = rng.normal(size=256)
    series_b = rng.normal(size=256)
    value = mutual_information(series_a, series_b)
    assert value == pytest.approx(0.0, abs=0.05)


def test_identical_series_are_above_independent():
    rng = np.random.default_rng(123)
    series = rng.normal(size=256)
    other = rng.normal(size=256)
    same = mutual_information(series, series)
    independent = mutual_information(series, other)
    assert same > independent
    assert same > 1.0


def test_squared_pair_is_above_independent():
    rng = np.random.default_rng(123)
    series_a = rng.normal(size=256)
    series_b = series_a * series_a
    other = rng.normal(size=256)
    nonlinear = mutual_information(series_a, series_b)
    independent = mutual_information(series_a, other)
    assert nonlinear > independent
    assert nonlinear > 1.0


def test_same_inputs_give_the_same_number():
    rng = np.random.default_rng(7)
    series_a = rng.normal(size=100)
    series_b = series_a * series_a
    first = mutual_information(series_a, series_b)
    second = mutual_information(series_a, series_b)
    assert first == second


def test_constant_series_fails_loudly():
    with pytest.raises(ValueError, match="constant"):
        mutual_information([1.0, 1.0, 1.0], [2.0, 3.0, 4.0])


def test_nan_fails_loudly():
    with pytest.raises(ValueError, match="nan or inf"):
        mutual_information([1.0, np.nan, 3.0], [2.0, 3.0, 4.0])


def test_length_mismatch_fails_loudly():
    with pytest.raises(ValueError, match="same length"):
        mutual_information([1.0, 2.0, 3.0], [1.0, 2.0])


def test_build_mi_graph_shape_and_symmetry():
    electrical_1s, electrical_4s, movement_4s = _synthetic_windows()
    graph = build_mi_graph(electrical_1s, electrical_4s, movement_4s)
    assert graph.shape == (N_NODES, N_NODES)
    np.testing.assert_allclose(graph, graph.T)
    for node_index in range(N_NODES):
        assert graph[node_index, node_index] > 0.0


def test_mi_uses_the_same_pair_series_as_pearson():
    electrical_1s, electrical_4s, movement_4s = _synthetic_windows()
    series_a, series_b = series_for_pair(0, 1, electrical_1s, electrical_4s, movement_4s)
    graph = build_mi_graph(electrical_1s, electrical_4s, movement_4s)
    expected = mutual_information(series_a, series_b)
    assert graph[0, 1] == pytest.approx(expected)
    assert len(series_a) == ELECTRICAL_SAMPLES_1S


def _synthetic_windows():
    rng = np.random.default_rng(123)
    electrical_1s = rng.normal(size=(N_ELECTRICAL_NODES, ELECTRICAL_SAMPLES_1S))
    electrical_4s = rng.normal(size=(N_ELECTRICAL_NODES, ELECTRICAL_SAMPLES_4S))
    movement_4s = rng.normal(size=(N_MOVEMENT_NODES, MOVEMENT_SAMPLES_4S))
    return electrical_1s, electrical_4s, movement_4s
