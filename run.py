import os

from edf import edf_gap_spans, load_run
from events import impd_spans, load_event_rows, seizure_spans
from glue import build_windows, keep_labelled_windows
from windows import TRAIN_STRIDE_SECONDS


def labelled_windows_from_files(
    eeg_path,
    ecg_path,
    emg_path,
    mov_path,
    events_path,
    stride_seconds=TRAIN_STRIDE_SECONDS,
):
    electrical, movement, _picked = load_run(
        eeg_path, ecg_path, emg_path, mov_path
    )
    gaps = []
    for path in (eeg_path, ecg_path, emg_path, mov_path):
        gaps.extend(edf_gap_spans(path))
    if len(gaps) > 0:
        raise ValueError(
            "edf has silent gaps, clock and samples wont line up, "
            f"we got {gaps}"
        )
    if not os.path.isfile(events_path):
        raise ValueError(f"events file is missing: {events_path}")
    rows = load_event_rows(events_path)
    features, graphs, ends = build_windows(
        electrical,
        movement,
        stride_seconds=stride_seconds,
        drop_spans=impd_spans(rows),
    )
    return keep_labelled_windows(
        features, graphs, ends, seizure_spans(rows)
    )
