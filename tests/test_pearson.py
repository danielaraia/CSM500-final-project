import numpy as np
import pytest

from graphs.pearson import (
    ELECTRICAL_SAMPLES_1S,
    ELECTRICAL_SAMPLES_4S,
    MOVEMENT_SAMPLES_4S,
    N_ELECTRICAL_NODES,
    N_MOVEMENT_NODES,
    N_NODES,
    build_pearson_graph,
    count_short_buffer_drops,
    pair_touches_movement,
    pearson_correlation,
    resample_electrical_to_25hz,
    series_for_pair,
    steps_with_full_movement_buffer,
)


def test_identical_series_have_pearson_one():
    series = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert pearson_correlation(series, series) == pytest.approx(1.0)


def test_known_correlation_matches_hand_value():
    series_a = [1.0, 2.0, 3.0, 4.0]
    series_b = [2.0, 3.0, 5.0, 8.0]
    # i worked this out on paper
    # means 2.5 and 4.5
    # a minus mean: -1.5, -0.5, 0.5, 1.5
    # b minus mean: -2.5, -1.5, 0.5, 3.5
    # dots add to 10
    # squares add to 5 and 21
    # 10 / sqrt(105) = 0.9759000729485332
    assert pearson_correlation(series_a, series_b) == pytest.approx(10.0 / np.sqrt(105.0))


def test_constant_series_fails_loudly():
    with pytest.raises(ValueError, match="constant"):
        pearson_correlation([1.0, 1.0, 1.0], [2.0, 3.0, 4.0])


def test_electrical_only_pair_uses_256_samples():
    electrical_1s, electrical_4s, movement_4s = _synthetic_windows()
    series_a, series_b = series_for_pair(0, 1, electrical_1s, electrical_4s, movement_4s)
    assert len(series_a) == ELECTRICAL_SAMPLES_1S
    assert len(series_b) == ELECTRICAL_SAMPLES_1S


def test_movement_touching_pair_uses_100_samples_at_25hz():
    electrical_1s, electrical_4s, movement_4s = _synthetic_windows()
    series_a, series_b = series_for_pair(0, 4, electrical_1s, electrical_4s, movement_4s)
    assert pair_touches_movement(0, 4)
    assert len(series_a) == MOVEMENT_SAMPLES_4S
    assert len(series_b) == MOVEMENT_SAMPLES_4S
    np.testing.assert_allclose(series_b, movement_4s[0])
    resampled = resample_electrical_to_25hz(electrical_4s[0])
    np.testing.assert_allclose(series_a, resampled)


def test_two_movement_nodes_use_native_25hz_series():
    electrical_1s, electrical_4s, movement_4s = _synthetic_windows()
    series_a, series_b = series_for_pair(4, 5, electrical_1s, electrical_4s, movement_4s)
    assert len(series_a) == MOVEMENT_SAMPLES_4S
    assert len(series_b) == MOVEMENT_SAMPLES_4S
    np.testing.assert_allclose(series_a, movement_4s[0])
    np.testing.assert_allclose(series_b, movement_4s[1])


def test_build_pearson_graph_shape_and_symmetry():
    electrical_1s, electrical_4s, movement_4s = _synthetic_windows()
    graph = build_pearson_graph(electrical_1s, electrical_4s, movement_4s)
    assert graph.shape == (N_NODES, N_NODES)
    np.testing.assert_allclose(graph, graph.T)
    for node_index in range(N_NODES):
        assert graph[node_index, node_index] == pytest.approx(1.0)


def test_short_buffer_at_start_and_after_a_gap():
    # 12 good secs, one gap, then 6 good secs
    step_is_valid = [True] * 12 + [False] + [True] * 6
    has_full_buffer = steps_with_full_movement_buffer(step_is_valid)
    assert has_full_buffer[:3] == [False, False, False]
    assert has_full_buffer[3] is True
    assert has_full_buffer[11] is True
    assert has_full_buffer[12] is False
    assert has_full_buffer[13:16] == [False, False, False]
    assert has_full_buffer[16] is True
    assert count_short_buffer_drops(step_is_valid) == 6


def _synthetic_windows():
    rng = np.random.default_rng(123)
    electrical_1s = rng.normal(size=(N_ELECTRICAL_NODES, ELECTRICAL_SAMPLES_1S))
    electrical_4s = rng.normal(size=(N_ELECTRICAL_NODES, ELECTRICAL_SAMPLES_4S))
    movement_4s = rng.normal(size=(N_MOVEMENT_NODES, MOVEMENT_SAMPLES_4S))
    return electrical_1s, electrical_4s, movement_4s
