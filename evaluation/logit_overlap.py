import numpy as np

from evaluation.alarms import load_window_scores


def overlap_row(labels, logits):
    labels = np.asarray(labels).reshape(-1)
    logits = np.asarray(logits).reshape(-1)
    if labels.shape[0] != logits.shape[0]:
        raise ValueError(
            "labels and logits need the same number of windows, "
            f"we got {labels.shape[0]} and {logits.shape[0]}"
        )
    sop_vals = []
    inter_vals = []
    index = 0
    while index < labels.shape[0]:
        lab = int(labels[index])
        val = float(logits[index])
        if lab == 1:
            sop_vals.append(val)
        elif lab == 0:
            inter_vals.append(val)
        else:
            raise ValueError(f"class numbers have to be 0 or 1, we got {lab}")
        index += 1
    row = {
        "n_sop": int(len(sop_vals)),
        "sop_median": _median(sop_vals),
        "sop_max": _max_or_none(sop_vals),
        "n_interictal": int(len(inter_vals)),
        "interictal_median": _median(inter_vals),
        "interictal_max": _max_or_none(inter_vals),
        "frac_sop_at_or_under_interictal_max": None,
    }
    if len(sop_vals) > 0 and len(inter_vals) > 0:
        inter_max = row["interictal_max"]
        n_under = 0
        sop_index = 0
        while sop_index < len(sop_vals):
            if sop_vals[sop_index] <= inter_max:
                n_under += 1
            sop_index += 1
        row["frac_sop_at_or_under_interictal_max"] = float(n_under) / float(
            len(sop_vals)
        )
    return row


def overlap_from_pack_and_scores(pack_path, scores_path):
    pack = np.load(pack_path)
    labels = pack["labels"]
    if "ends" not in pack.files:
        raise ValueError(f"pack has no stored ends: {pack_path}")
    pack_ends = pack["ends"]
    ends, logits = load_window_scores(scores_path)
    if len(pack_ends) != len(ends):
        raise ValueError(
            "pack ends and score ends need the same number of windows, "
            f"we got {len(pack_ends)} and {len(ends)}"
        )
    index = 0
    while index < len(ends):
        if int(pack_ends[index]) != int(ends[index]):
            raise ValueError(
                "pack ends do not match score ends, "
                f"pack {int(pack_ends[index])} scores {int(ends[index])}"
            )
        index += 1
    return overlap_row(labels, logits)


def still_overlap(row):
    frac = row["frac_sop_at_or_under_interictal_max"]
    if frac is None:
        raise ValueError("cannot judge overlap with no sop or no interictal windows")
    return frac >= 1.0


def _median(values):
    if len(values) < 1:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return float(ordered[mid - 1] + ordered[mid]) / 2.0


def _max_or_none(values):
    if len(values) < 1:
        return None
    return float(max(values))
