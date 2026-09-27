from train.train_more_people import _patient_ids, packed_folder


def test_packed_folder_is_eval1_pearson():
    path = packed_folder("sub-015")
    assert path.endswith("sub-015/pearson/sop30/seed123/eval1")


def test_patient_ids_need_at_least_one():
    try:
        _patient_ids(["train_more_people.py"])
    except ValueError:
        pass
    else:
        raise AssertionError("empty argv should fail")
    assert _patient_ids(["train_more_people.py", "sub-015", "sub-030"]) == (
        "sub-015",
        "sub-030",
    )
