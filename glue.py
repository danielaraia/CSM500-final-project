import numpy as np

from features import log_amplitude_fft
from filters import filter_ecg, filter_eeg, filter_emg, filter_movement_axis
from labels import LABEL_UNUSED, label_second
from movement import axis_magnitude, upsample_movement_to_256
from pearson import (
    ELECTRICAL_SAMPLES_1S,
    MOVEMENT_BUFFER_SECONDS,
    MOVEMENT_SAMPLES_4S,
    N_ELECTRICAL_NODES,
    N_MOVEMENT_NODES,
    N_NODES,
    build_pearson_graph,
    steps_with_full_movement_buffer,
)
from quality import second_is_bad
from windows import (
    EVAL_STRIDE_SECONDS,
    TRAIN_STRIDE_SECONDS,
    WINDOW_SECONDS,
    window_end_times,
)

N_MOVEMENT_AXES = 12
AXES_PER_DEVICE = 3
SAMPLES_MOVEMENT_1S = 25
# mixed pearson pairs need 4 sec at 25 Hz, so graphs 0,1,2 dont exist
SECONDS_BEFORE_FIRST_GRAPH = MOVEMENT_BUFFER_SECONDS - 1
SECONDS_NEEDED = SECONDS_BEFORE_FIRST_GRAPH + WINDOW_SECONDS


def build_one_window(electrical, movement_axes):
    features, graphs, _ends = build_windows(
        electrical, movement_axes, stride_seconds=EVAL_STRIDE_SECONDS
    )
    if features.shape[0] < 1:
        raise ValueError("didnt get a window, too many bad seconds")
    return features[0], graphs[0]


def build_windows(electrical, movement_axes, stride_seconds=TRAIN_STRIDE_SECONDS):
    electrical = np.asarray(electrical, dtype=np.float64)
    movement_axes = np.asarray(movement_axes, dtype=np.float64)
    _n_seconds(electrical, movement_axes)

    electrical = _filter_electrical(electrical)
    movement = _movement_magnitudes(movement_axes)
    electrical_seconds = _cut_seconds(electrical, ELECTRICAL_SAMPLES_1S)
    movement_seconds = _cut_seconds(movement, SAMPLES_MOVEMENT_1S)

    quality_ok = _quality_ok_seconds(electrical_seconds, movement_seconds)
    graph_ok = _graph_ok_seconds(quality_ok)
    ends = window_end_times(graph_ok, WINDOW_SECONDS, stride_seconds)

    n_windows = len(ends)
    features = np.zeros((n_windows, WINDOW_SECONDS, N_NODES, 128), dtype=np.float64)
    adjacency = np.zeros((n_windows, WINDOW_SECONDS, N_NODES, N_NODES), dtype=np.float64)
    feat_by_second = {}
    graph_by_second = {}
    window_index = 0
    while window_index < n_windows:
        end_time = ends[window_index]
        start_time = end_time - WINDOW_SECONDS + 1
        step = 0
        while step < WINDOW_SECONDS:
            index = start_time + step
            if index not in feat_by_second:
                if not graph_ok[index]:
                    raise ValueError(
                        f"window end {end_time} wants sec {index} "
                        f"bjt that second isnt graph ok"
                    )
                feat_by_second[index] = _features_for_second(
                    electrical_seconds[index],
                    movement_seconds[index],
                )
                graph_by_second[index] = _pearson_for_second(
                    electrical_seconds, movement_seconds, index
                )
            features[window_index, step] = feat_by_second[index]
            adjacency[window_index, step] = graph_by_second[index]
            step += 1
        window_index += 1
    return features, adjacency, ends


def keep_labelled_windows(features, adjacency, ends, seizures):
    features = np.asarray(features)
    adjacency = np.asarray(adjacency)
    if features.shape[0] != adjacency.shape[0] or features.shape[0] != len(ends):
        raise ValueError(
            "features, graphs and ends need the same count, "
            f"we got {features.shape[0]}, {adjacency.shape[0]}, {len(ends)}"
        )
    kept_features = []
    kept_graphs = []
    kept_ends = []
    kept_labels = []
    index = 0
    while index < len(ends):
        # label sits on the last second of the window, not the whole 60
        lab = label_second(ends[index], seizures)
        if lab != LABEL_UNUSED:
            kept_features.append(features[index])
            kept_graphs.append(adjacency[index])
            kept_ends.append(ends[index])
            kept_labels.append(lab)
        index += 1
    if len(kept_labels) == 0:
        raise ValueError("didnt keep any labelled windows")
    return (
        np.stack(kept_features, axis=0),
        np.stack(kept_graphs, axis=0),
        kept_ends,
        kept_labels,
    )


def _n_seconds(electrical, movement_axes):
    if electrical.ndim != 2 or electrical.shape[0] != N_ELECTRICAL_NODES:
        raise ValueError(
            f"electrical should be ({N_ELECTRICAL_NODES}, samples), we got {electrical.shape}"
        )
    if movement_axes.ndim != 2 or movement_axes.shape[0] != N_MOVEMENT_AXES:
        raise ValueError(
            f"movement should be ({N_MOVEMENT_AXES}, samples), we got {movement_axes.shape}"
        )
    if electrical.shape[1] % ELECTRICAL_SAMPLES_1S != 0:
        raise ValueError(
            f"electrical length has to split into 1 sec chunks, we got {electrical.shape[1]}"
        )
    if movement_axes.shape[1] % SAMPLES_MOVEMENT_1S != 0:
        raise ValueError(
            f"movement length has to split into 1 sec chunks, we got {movement_axes.shape[1]}"
        )
    n_electrical = electrical.shape[1] // ELECTRICAL_SAMPLES_1S
    n_movement = movement_axes.shape[1] // SAMPLES_MOVEMENT_1S
    if n_electrical != n_movement:
        raise ValueError(
            f"electrical and movement need the same number of seconds, "
            f"we got {n_electrical} and {n_movement}"
        )
    return n_electrical


def _quality_ok_seconds(electrical_seconds, movement_seconds):
    n_seconds = electrical_seconds.shape[0]
    ok = []
    second_index = 0
    while second_index < n_seconds:
        bad = second_is_bad(
            electrical_seconds[second_index, 0],
            electrical_seconds[second_index, 1],
            electrical_seconds[second_index, 2],
            electrical_seconds[second_index, 3],
            movement_seconds[second_index, 0],
            movement_seconds[second_index, 1],
            movement_seconds[second_index, 2],
            movement_seconds[second_index, 3],
        )
        ok.append(not bad)
        second_index += 1
    return ok


def _graph_ok_seconds(quality_ok):
    has_buffer = steps_with_full_movement_buffer(quality_ok)
    graph_ok = []
    second_index = 0
    while second_index < len(quality_ok):
        graph_ok.append(bool(quality_ok[second_index] and has_buffer[second_index]))
        second_index += 1
    return graph_ok


def _filter_electrical(electrical):
    out = np.zeros_like(electrical)
    out[0] = filter_eeg(electrical[0])
    out[1] = filter_eeg(electrical[1])
    out[2] = filter_ecg(electrical[2])
    out[3] = filter_emg(electrical[3])
    return out


def _movement_magnitudes(movement_axes):
    magnitudes = np.zeros(
        (N_MOVEMENT_NODES, movement_axes.shape[1]), dtype=np.float64
    )
    device = 0
    while device < N_MOVEMENT_NODES:
        start = device * AXES_PER_DEVICE
        x = filter_movement_axis(movement_axes[start])
        y = filter_movement_axis(movement_axes[start + 1])
        z = filter_movement_axis(movement_axes[start + 2])
        magnitudes[device] = axis_magnitude(x, y, z)
        device += 1
    return magnitudes


def _cut_seconds(run, samples_per_second):
    n_seconds = run.shape[1] // samples_per_second
    n_nodes = run.shape[0]
    seconds = np.zeros((n_seconds, n_nodes, samples_per_second), dtype=np.float64)
    second_index = 0
    while second_index < n_seconds:
        start = second_index * samples_per_second
        seconds[second_index] = run[:, start : start + samples_per_second]
        second_index += 1
    return seconds


def _features_for_second(electrical_1s, movement_1s):
    feats = np.zeros((N_NODES, 128), dtype=np.float64)
    node = 0
    while node < N_ELECTRICAL_NODES:
        feats[node] = log_amplitude_fft(electrical_1s[node])
        node += 1
    node = 0
    while node < N_MOVEMENT_NODES:
        up = upsample_movement_to_256(movement_1s[node])
        feats[N_ELECTRICAL_NODES + node] = log_amplitude_fft(up)
        node += 1
    return feats


def _pearson_for_second(electrical_seconds, movement_seconds, second_index):
    electrical_1s = electrical_seconds[second_index]
    electrical_chunks = []
    movement_chunks = []
    offset = 0
    while offset < MOVEMENT_BUFFER_SECONDS:
        index = second_index - SECONDS_BEFORE_FIRST_GRAPH + offset
        electrical_chunks.append(electrical_seconds[index])
        movement_chunks.append(movement_seconds[index])
        offset += 1
    electrical_4s = np.concatenate(electrical_chunks, axis=1)
    movement_4s = np.concatenate(movement_chunks, axis=1)
    if movement_4s.shape[1] != MOVEMENT_SAMPLES_4S:
        raise ValueError(
            f"4 sec movement should be {MOVEMENT_SAMPLES_4S} samples, we got {movement_4s.shape[1]}"
        )
    return build_pearson_graph(electrical_1s, electrical_4s, movement_4s)
