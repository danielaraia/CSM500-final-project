import numpy as np
import pytest

from data.features import N_FFT_BINS
from model.glue import (
    SECONDS_NEEDED,
    build_one_window,
    build_windows,
    cut_monitoring_slice,
    keep_labelled_windows,
    split_labelled_windows,
)
from data.labels import LABEL_INTERICTAL, LABEL_SOP, LABEL_UNUSED, label_second
from graphs.pearson import N_NODES
from data.windows import EVAL_STRIDE_SECONDS, TRAIN_STRIDE_SECONDS, WINDOW_SECONDS


WIDE_RAILS = [
    (-1000.0, 1000.0),
    (-1000.0, 1000.0),
    (-1000.0, 1000.0),
    (-1000.0, 1000.0),
]


def test_fake_run_makes_one_sixty_step_window():
    features, graphs = _fake_window()
    assert features.shape == (WINDOW_SECONDS, N_NODES, N_FFT_BINS)
    assert graphs.shape == (WINDOW_SECONDS, N_NODES, N_NODES)
    # 10 Hz sine, dc dropped, so bin 10 is index 9
    peak_index = int(np.argmax(features[30, 0]))
    assert peak_index == 9
    np.testing.assert_allclose(graphs[30], graphs[30].T)
    assert graphs[30, 0, 1] == pytest.approx(1.0)
    node = 0
    while node < N_NODES:
        assert graphs[30, node, node] == pytest.approx(1.0)
        node += 1


def test_clean_eighty_seconds_has_two_train_windows():
    electrical, movement = _fake_run(80)
    features, graphs, ends, drop_counts = build_windows(
        electrical, movement, electrical_rails=WIDE_RAILS
    )
    # first 3 sec have no pearson buffer, so end 59 is a hole, 69 and 79 survive
    assert ends == [69, 79]
    assert drop_counts["n_steps_dropped_short_buffer"] == 3
    assert drop_counts["n_steps_dropped_qc"] == 0
    assert features.shape == (2, WINDOW_SECONDS, N_NODES, N_FFT_BINS)
    assert graphs.shape == (2, WINDOW_SECONDS, N_NODES, N_NODES)


def test_hole_at_fifteen_drops_the_early_train_window():
    electrical, movement = _fake_run(80)
    start = 15 * 256
    electrical[0, start : start + 256] = 0.0
    electrical[1, start : start + 256] = 0.0
    features, graphs, ends, drop_counts = build_windows(
        electrical, movement, electrical_rails=WIDE_RAILS
    )
    assert ends == [79]
    assert features.shape[0] == 1
    assert graphs.shape[0] == 1
    assert drop_counts["n_steps_dropped_qc"] >= 1


def test_impd_span_drops_the_early_train_window():
    electrical, movement = _fake_run(80)
    features, graphs, ends, drop_counts = build_windows(
        electrical, movement, drop_spans=[(15.0, 16.0)], electrical_rails=WIDE_RAILS
    )
    assert ends == [79]
    assert features.shape[0] == 1
    assert drop_counts["n_steps_qc_drop_span"] == 1


def test_missing_rails_fail():
    electrical, movement = _fake_run(80)
    with pytest.raises(ValueError, match="need converter rails"):
        build_windows(electrical, movement)


def test_raw_rail_hit_drops_the_early_train_window():
    electrical, movement = _fake_run(80)
    electrical[0, 15 * 256] = 1000.0
    tight_rails = [(-200.0, 200.0), (-200.0, 200.0), (-200.0, 200.0), (-200.0, 200.0)]
    features, graphs, ends, drop_counts = build_windows(
        electrical, movement, electrical_rails=tight_rails
    )
    assert ends == [79]
    assert features.shape[0] == 1
    assert graphs.shape[0] == 1
    assert drop_counts["n_steps_qc_electrical_rail"] >= 1


def test_too_short_fails():
    electrical = np.ones((4, 256))
    movement = np.ones((12, 25))
    with pytest.raises(ValueError, match="didnt get a window"):
        build_one_window(electrical, movement, WIDE_RAILS)


def test_second_count_mismatch_fails():
    electrical = np.ones((4, 256 * 4))
    movement = np.ones((12, 25 * 3))
    with pytest.raises(ValueError, match="we got 4 and 3"):
        build_one_window(electrical, movement, WIDE_RAILS)


def test_keep_labelled_drops_sph_and_keeps_sop():
    features = np.zeros((2, WINDOW_SECONDS, N_NODES, N_FFT_BINS))
    graphs = np.zeros((2, WINDOW_SECONDS, N_NODES, N_NODES))
    features[0, 0, 0, 0] = 1.0
    ends = [69, 79]
    # onset 375: SOP ends at 75, SPH is 75 to 375, so 69 is sop and 79 is unused
    seizures = [(375, 400)]
    kept_f, kept_g, kept_ends, kept_labels = keep_labelled_windows(
        features, graphs, ends, seizures
    )
    assert kept_ends == [69]
    assert kept_labels == [LABEL_SOP]
    assert kept_f.shape[0] == 1
    assert kept_f[0, 0, 0, 0] == 1.0
    assert label_second(79, seizures) == LABEL_UNUSED


def test_all_unused_windows_fail():
    features = np.zeros((2, WINDOW_SECONDS, N_NODES, N_FFT_BINS))
    graphs = np.zeros((2, WINDOW_SECONDS, N_NODES, N_NODES))
    ends = [69, 79]
    seizures = [(80, 100)]
    with pytest.raises(ValueError, match="didnt keep any labelled windows"):
        keep_labelled_windows(features, graphs, ends, seizures)


def test_split_puts_windows_before_t1_in_train():
    features = np.zeros((4, WINDOW_SECONDS, N_NODES, N_FFT_BINS))
    graphs = np.zeros((4, WINDOW_SECONDS, N_NODES, N_NODES))
    features[0, 0, 0, 0] = 1.0
    features[3, 0, 0, 0] = 9.0
    ends = [10, 20, 50, 80]
    labels = [LABEL_SOP, LABEL_SOP, LABEL_SOP, LABEL_SOP]
    train, val, test = split_labelled_windows(
        features, graphs, ends, labels, t1=30, t2=60
    )
    train_f, _train_g, train_ends, _train_l = train
    val_f, _val_g, val_ends, _val_l = val
    test_f, _test_g, test_ends, _test_l = test
    assert train_ends == [10, 20]
    assert val_ends == [50]
    assert test_ends == [80]
    assert train_f[0, 0, 0, 0] == 1.0
    assert test_f[0, 0, 0, 0] == 9.0
    assert val_f.shape[0] == 1


def test_split_fails_without_train_windows():
    features = np.zeros((2, WINDOW_SECONDS, N_NODES, N_FFT_BINS))
    graphs = np.zeros((2, WINDOW_SECONDS, N_NODES, N_NODES))
    ends = [50, 80]
    labels = [LABEL_SOP, LABEL_SOP]
    with pytest.raises(ValueError, match="no train windows"):
        split_labelled_windows(features, graphs, ends, labels, t1=30, t2=60)


def test_monitoring_cut_keeps_last_quarter():
    n_windows = 16
    features = np.zeros((n_windows, WINDOW_SECONDS, N_NODES, N_FFT_BINS))
    graphs = np.zeros((n_windows, WINDOW_SECONDS, N_NODES, N_NODES))
    features[0, 0, 0, 0] = 1.0
    features[15, 0, 0, 0] = 9.0
    ends = list(range(n_windows))
    labels = [LABEL_INTERICTAL] * 12 + [LABEL_SOP] * 2 + [LABEL_INTERICTAL] * 2
    train, monitor, monitor_start, n_positive, n_negative = cut_monitoring_slice(
        features, graphs, ends, labels
    )
    train_f, _g, train_ends, train_labels = train
    mon_f, _mg, mon_ends, mon_labels = monitor
    assert monitor_start == 12
    assert n_positive == 2
    assert n_negative == 2
    assert train_ends == list(range(12))
    assert mon_ends == [12, 13, 14, 15]
    assert train_f.shape[0] == 12
    assert mon_f[3, 0, 0, 0] == 9.0
    assert train_labels[-1] == LABEL_INTERICTAL
    assert mon_labels[0] == LABEL_SOP


def test_pack_keeps_train_hop_ten_and_eval_hop_one():
    electrical, movement = _fake_run(180)
    # far seizure: seconds 0 to 180 are all interictal
    seizures = [(100000, 100100)]
    features, graphs, ends, drop_counts = build_windows(
        electrical,
        movement,
        stride_seconds=EVAL_STRIDE_SECONDS,
        electrical_rails=WIDE_RAILS,
        seizures=seizures,
        t1=100,
        train_stride_seconds=TRAIN_STRIDE_SECONDS,
    )
    assert drop_counts["n_steps_dropped_qc"] == 0
    assert 62 not in ends
    assert 69 in ends
    assert 70 not in ends
    assert 79 in ends
    assert 89 in ends
    assert 99 in ends
    assert 100 in ends
    assert 101 in ends
    assert 179 in ends
    train, val, test = split_labelled_windows(
        features,
        graphs,
        ends,
        [LABEL_INTERICTAL] * len(ends),
        t1=100,
        t2=150,
    )
    assert train[2] == [69, 79, 89, 99]
    assert val[2][0] == 100
    assert val[2][1] == 101
    assert val[2][-1] == 149
    assert test[2][0] == 150
    assert test[2][-1] == 179


def test_pack_keeps_hop_one_inside_monitor_span():
    electrical, movement = _fake_run(250)
    seizures = [(100000, 100100)]
    features, graphs, ends, drop_counts = build_windows(
        electrical,
        movement,
        stride_seconds=EVAL_STRIDE_SECONDS,
        electrical_rails=WIDE_RAILS,
        seizures=seizures,
        t1=200,
        train_stride_seconds=TRAIN_STRIDE_SECONDS,
    )
    assert drop_counts["n_steps_dropped_qc"] == 0
    # hop 10 train before the last quarter: 69 through 169
    assert 169 in ends
    assert 170 not in ends
    assert 178 not in ends
    # last 25 percent of hop 10 train starts at 179, packed at hop 1
    assert 179 in ends
    assert 180 in ends
    assert 181 in ends
    assert 199 in ends
    train, val, test = split_labelled_windows(
        features,
        graphs,
        ends,
        [LABEL_INTERICTAL] * len(ends),
        t1=200,
        t2=220,
    )
    assert 170 not in train[2]
    assert 179 in train[2]
    assert 180 in train[2]
    assert 199 in train[2]
    assert val[2][0] == 200
    assert test[2][0] == 220


def test_t1_without_train_stride_fails():
    electrical, movement = _fake_run(80)
    with pytest.raises(ValueError, match="t1 and train stride"):
        build_windows(
            electrical,
            movement,
            electrical_rails=WIDE_RAILS,
            t1=70,
        )


def _fake_window():
    electrical, movement = _fake_run(SECONDS_NEEDED)
    return build_one_window(electrical, movement, WIDE_RAILS)


def _fake_run(n_sec):
    time = np.arange(n_sec * 256) / 256.0
    eeg = 80.0 * np.sin(2 * np.pi * 10.0 * time)
    emg = 80.0 * np.sin(2 * np.pi * 40.0 * time)
    electrical = np.stack([eeg, eeg, eeg, emg])
    rng = np.random.default_rng(123)
    movement = rng.normal(size=(12, n_sec * 25))
    return electrical, movement
