import numpy as np

from tools.label_audit import audit_packed_labels, oracle_label


def test_oracle_keeps_postictal_time_unused():
    seizures = [(10000, 10100), (15000, 15100)]
    assert oracle_label(13000, seizures, 1800) is None
    assert oracle_label(13700, seizures, 1800) == 1


def test_audit_finds_a_wrong_positive(tmp_path):
    np.savez_compressed(
        tmp_path / "train.npz",
        labels=np.asarray([1, 1]),
        ends=np.asarray([13000, 13700]),
    )
    result = audit_packed_labels(
        str(tmp_path), [(10000, 10100), (15000, 15100)], 1800
    )
    assert result["n_checked"] == 2
    assert result["n_mistakes"] == 1
    assert result["all_labels_match"] is False
