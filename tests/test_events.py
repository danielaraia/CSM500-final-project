import os
import tempfile

import pytest

from data.events import impd_spans, load_event_rows, seizure_spans, unique_event_types


def test_seizure_and_impd_from_a_tiny_tsv():
    text = (
        "onset\tduration\teventType\n"
        "10.0\t30.0\tsz\n"
        "100.0\t5.0\timpd\n"
        "0.0\t200.0\tbckg\n"
    )
    path = _write_tsv(text)
    try:
        rows = load_event_rows(path)
        assert unique_event_types(rows) == ["sz", "impd", "bckg"]
        assert seizure_spans(rows) == [(10.0, 40.0)]
        assert impd_spans(rows) == [(100.0, 105.0)]
    finally:
        os.remove(path)


def test_mystery_impedance_name_fails():
    text = "onset\tduration\teventType\n0.0\t1.0\timpedance_check\n"
    path = _write_tsv(text)
    try:
        rows = load_event_rows(path)
        with pytest.raises(ValueError, match="impedance"):
            impd_spans(rows)
    finally:
        os.remove(path)


def test_seizure_without_duration_fails():
    text = "onset\tduration\teventType\n10.0\t0\tsz\n"
    path = _write_tsv(text)
    try:
        rows = load_event_rows(path)
        with pytest.raises(ValueError, match="no duration"):
            seizure_spans(rows)
    finally:
        os.remove(path)


def _write_tsv(text):
    handle = tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False)
    handle.write(text)
    handle.close()
    return handle.name
