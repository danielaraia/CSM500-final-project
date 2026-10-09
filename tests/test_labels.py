import pytest

from data.labels import (
    INTERICTAL_GAP_SECONDS,
    LABEL_INTERICTAL,
    LABEL_SOP,
    LABEL_UNUSED,
    SOP_SECONDS,
    SPH_SECONDS,
    label_second,
    monitoring_indices,
    n_test_seizures,
    seizures_with_full_sop,
    split_cut_times,
)


def test_sop_sph_ictal_and_interictal_on_known_clock():
    onset = 10_000
    offset = 10_200
    seizures = [(onset, offset)]
    sop_start = onset - SPH_SECONDS - SOP_SECONDS
    sph_start = onset - SPH_SECONDS

    assert label_second(sop_start, seizures) == LABEL_SOP
    assert label_second(sph_start - 1, seizures) == LABEL_SOP
    assert label_second(sph_start, seizures) == LABEL_UNUSED
    assert label_second(onset - 1, seizures) == LABEL_UNUSED
    assert label_second(onset, seizures) == LABEL_UNUSED
    assert label_second(offset - 1, seizures) == LABEL_UNUSED
    assert label_second(offset, seizures) == LABEL_UNUSED
    assert label_second(offset + INTERICTAL_GAP_SECONDS - 1, seizures) == LABEL_UNUSED
    assert label_second(offset + INTERICTAL_GAP_SECONDS, seizures) == LABEL_INTERICTAL
    assert label_second(onset - INTERICTAL_GAP_SECONDS - 1, seizures) == LABEL_INTERICTAL
    assert label_second(onset - INTERICTAL_GAP_SECONDS, seizures) == LABEL_UNUSED


def test_postictal_period_overrides_the_next_seizure_sop():
    seizures = [
        (10_000, 10_100),
        (15_000, 15_100),
    ]

    assert label_second(13_000, seizures) == LABEL_UNUSED
    assert label_second(13_699, seizures) == LABEL_UNUSED
    assert label_second(13_700, seizures) == LABEL_SOP


def test_one_usable_seizure_is_rejected():
    with pytest.raises(ValueError, match="at least two usable seizures"):
        n_test_seizures(1)


def test_early_seizure_is_dropped_without_a_full_sop():
    seizures = [
        (1359.0, 1401.0),
        (8300.0, 8381.0),
        (19956.0, 20062.0),
        (26559.0, 26644.0),
    ]
    kept = seizures_with_full_sop(seizures)
    assert kept == [
        (8300.0, 8381.0),
        (19956.0, 20062.0),
        (26559.0, 26644.0),
    ]


def test_three_seizures_put_one_in_test_and_cut_at_gap_midpoint():
    seizures = [
        (10_000, 10_100),
        (30_000, 30_100),
        (50_000, 50_100),
    ]
    assert n_test_seizures(3) == 1
    t1, t2, n_train, n_test = split_cut_times(seizures)
    assert n_train == 2
    assert n_test == 1
    assert t2 == pytest.approx((30_100 + 50_000) / 2.0)
    assert t1 <= t2
    assert t1 >= 30_100


def test_monitoring_slice_fails_when_one_class_is_missing():
    labels = [LABEL_INTERICTAL] * 6 + [LABEL_SOP] * 2
    with pytest.raises(ValueError, match="missing a class"):
        monitoring_indices(labels)


def test_monitoring_slice_is_last_quarter_with_both_classes():
    # last 25% of train labels, in time order
    labels = [LABEL_INTERICTAL] * 12 + [LABEL_SOP] * 2 + [LABEL_INTERICTAL] * 2
    monitor_start, n_positive, n_negative = monitoring_indices(labels)
    assert monitor_start == 12
    assert n_positive == 2
    assert n_negative == 2
