import os

from tools.label_audit import audit_one, sop_second_counts_by_region

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_sop_seconds_split_across_regions():
    # one seizure, onset 4000; sop window at 30 min is [1900, 3700)
    seizures = [(4000.0, 4100.0)]
    sop_seconds = 30 * 60
    counts = sop_second_counts_by_region(seizures, sop_seconds, t1=2000, t2=3000)
    assert counts["train"] == 100  # [1900, 2000)
    assert counts["val"] == 1000  # [2000, 3000)
    assert counts["test"] == 700  # [3000, 3700)
    assert sum(counts.values()) == 1800


def test_no_sop_seconds_when_onset_is_before_sph_end():
    # onset 250 is inside the sph: the sop window ends before second 0, no sop seconds at all
    seizures = [(250.0, 300.0)]
    counts = sop_second_counts_by_region(seizures, 30 * 60, t1=100, t2=200)
    assert counts == {"train": 0, "val": 0, "test": 0}


def test_partial_sop_window_is_still_labelled():
    # onset 500: full 30 min sop does not fit, but the partial [0, 200) does get
    # labelled sop (the usability filter only affects the split, not labelling)
    seizures = [(500.0, 560.0)]
    counts = sop_second_counts_by_region(seizures, 30 * 60, t1=100, t2=200)
    assert counts["train"] == 100  # [0, 100)
    assert counts["val"] == 100  # [100, 200)
    assert counts["test"] == 0


def test_audit_one_recomputes_locked_split_for_a_real_pack():
    # integration check against a known-good pack, if its local data is present
    metrics = os.path.join(
        ROOT,
        "experiments/runs/within_patient/sub-051/pearson/sop15/seed123/eval1/metrics.json",
    )
    events = os.path.join(
        ROOT, "data/seizeit2/sub-051/ses-01/eeg/sub-051_ses-01_task-szMonitoring_run-01_events.tsv"
    )
    if not (os.path.isfile(metrics) and os.path.isfile(events)):
        return
    result = audit_one(ROOT, "sub-051", 15)
    assert result["status"] == "audited"
    assert result["checks"]["t1"] is True
    assert result["checks"]["t2"] is True
    assert result["checks"]["n_train_seizures"] is True
    assert result["checks"]["n_test_seizures"] is True
    assert result["all_checks_pass"] is True
