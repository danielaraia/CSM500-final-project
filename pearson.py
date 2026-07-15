import numpy as np
from scipy import signal

N_ELECTRICAL_NODES = 4
N_MOVEMENT_NODES = 4
N_NODES = N_ELECTRICAL_NODES + N_MOVEMENT_NODES
ELECTRICAL_SAMPLES_1S = 256
ELECTRICAL_SAMPLES_4S = 1024
MOVEMENT_SAMPLES_4S = 100
MOVEMENT_BUFFER_SECONDS = 4


def pearson_correlation(series_a, series_b):
    if len(series_a) != len(series_b):
        raise ValueError(
            "A Pearson needs the two series of the same length, "
            f"bjt we got {len(series_a)} and {len(series_b)}"
        )
    if len(series_a) < 2:
        raise ValueError("need at least 2 samples or we cant do pearson")

    a = np.asarray(series_a, dtype=np.float64)
    b = np.asarray(series_b, dtype=np.float64)
    a_centered = a - np.mean(a)
    b_centered = b - np.mean(b)
    numerator = np.sum(a_centered * b_centered)
    denominator = np.sqrt(np.sum(a_centered * a_centered) * np.sum(b_centered * b_centered))
    if denominator == 0.0:
        raise ValueError("cant do pearson, a series is constant")
    return float(numerator / denominator)


def pair_touches_movement(node_index_a, node_index_b):
    if node_index_a < 0 or node_index_b < 0:
        raise ValueError("cant have a negative node")
    if node_index_a >= N_NODES or node_index_b >= N_NODES:
        raise ValueError(f"node has to be under {N_NODES}, we only have {N_NODES}")
    return node_index_a >= N_ELECTRICAL_NODES or node_index_b >= N_ELECTRICAL_NODES


def resample_electrical_to_25hz(electrical_four_seconds):
    samples = np.asarray(electrical_four_seconds, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError("this needs to be 1d")
    if samples.shape[0] != ELECTRICAL_SAMPLES_4S:
        raise ValueError(
            "4 sec electrical should be "
            f"{ELECTRICAL_SAMPLES_4S} samples at 256 Hz, we got {samples.shape[0]}"
        )
    # 256 into 25 doesnt go evenly, so taking every 10th sample is the wrong rate.
    # same 4 sec window, just 100 points.
    return signal.resample(samples, MOVEMENT_SAMPLES_4S)


def series_for_pair(
    node_index_a,
    node_index_b,
    electrical_one_second,
    electrical_four_seconds,
    movement_four_seconds,
):
    if pair_touches_movement(node_index_a, node_index_b):
        series_a = _movement_rule_series(
            node_index_a, electrical_four_seconds, movement_four_seconds
        )
        series_b = _movement_rule_series(
            node_index_b, electrical_four_seconds, movement_four_seconds
        )
        return series_a, series_b

    series_a = _electrical_one_second_series(node_index_a, electrical_one_second)
    series_b = _electrical_one_second_series(node_index_b, electrical_one_second)
    return series_a, series_b


def build_pearson_graph(
    electrical_one_second,
    electrical_four_seconds,
    movement_four_seconds,
):
    electrical_1s = _require_shape(
        electrical_one_second, (N_ELECTRICAL_NODES, ELECTRICAL_SAMPLES_1S), "electrical_one_second"
    )
    electrical_4s = _require_shape(
        electrical_four_seconds,
        (N_ELECTRICAL_NODES, ELECTRICAL_SAMPLES_4S),
        "electrical_four_seconds",
    )
    movement_4s = _require_shape(
        movement_four_seconds, (N_MOVEMENT_NODES, MOVEMENT_SAMPLES_4S), "movement_four_seconds"
    )

    graph = np.zeros((N_NODES, N_NODES), dtype=np.float64)
    for node_a in range(N_NODES):
        for node_b in range(node_a, N_NODES):
            series_a, series_b = series_for_pair(
                node_a, node_b, electrical_1s, electrical_4s, movement_4s
            )
            value = pearson_correlation(series_a, series_b)
            graph[node_a, node_b] = value
            graph[node_b, node_a] = value
    return graph


def steps_with_full_movement_buffer(step_is_valid, buffer_seconds=MOVEMENT_BUFFER_SECONDS):
    if buffer_seconds < 1:
        raise ValueError("buffer_seconds has to be 1 or more")
    n_steps = len(step_is_valid)
    has_full_buffer = []
    for step_index in range(n_steps):
        if step_index + 1 < buffer_seconds:
            has_full_buffer.append(False)
            continue
        window_start = step_index - buffer_seconds + 1
        window_is_full = True
        for offset in range(buffer_seconds):
            if not step_is_valid[window_start + offset]:
                window_is_full = False
                break
        has_full_buffer.append(window_is_full)
    return has_full_buffer


def count_short_buffer_drops(step_is_valid, buffer_seconds=MOVEMENT_BUFFER_SECONDS):
    has_full_buffer = steps_with_full_movement_buffer(step_is_valid, buffer_seconds)
    n_dropped = 0
    for step_index in range(len(step_is_valid)):
        if step_is_valid[step_index] and not has_full_buffer[step_index]:
            n_dropped += 1
    return n_dropped


def _electrical_one_second_series(node_index, electrical_one_second):
    if node_index >= N_ELECTRICAL_NODES:
        raise ValueError("electrical 1s rule is only for the eeg/ecg/emg nodes")
    return np.asarray(electrical_one_second[node_index], dtype=np.float64)


def _movement_rule_series(node_index, electrical_four_seconds, movement_four_seconds):
    if node_index < N_ELECTRICAL_NODES:
        return resample_electrical_to_25hz(electrical_four_seconds[node_index])
    movement_index = node_index - N_ELECTRICAL_NODES
    series = np.asarray(movement_four_seconds[movement_index], dtype=np.float64)
    if series.shape[0] != MOVEMENT_SAMPLES_4S:
        raise ValueError(
            "movement 4 sec should be "
            f"{MOVEMENT_SAMPLES_4S} samples at 25 Hz, we got {series.shape[0]}"
        )
    return series


def _require_shape(array, expected_shape, name):
    values = np.asarray(array, dtype=np.float64)
    if values.shape != expected_shape:
        raise ValueError(f"{name} should be {expected_shape}, we got {values.shape}")
    return values
