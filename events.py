import csv


def load_event_rows(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("events file has no header")
        if "eventType" not in reader.fieldnames:
            raise ValueError("events file is missing eventType")
        if "onset" not in reader.fieldnames:
            raise ValueError("events file is missing onset")
        rows = []
        for row in reader:
            rows.append(row)
        return rows


def unique_event_types(rows):
    names = []
    for row in rows:
        name = row["eventType"].strip()
        if name not in names:
            names.append(name)
    return names


def seizure_onsets(rows):
    onsets = []
    for row in rows:
        if row["eventType"].strip().startswith("sz"):
            # sz and sz_whatever both count as a seizure
            onsets.append(float(row["onset"]))
    return onsets


def impd_spans(rows):
    _fail_if_mystery_impedance(rows)
    spans = []
    for row in rows:
        if row["eventType"].strip() == "impd":
            start = float(row["onset"])
            duration = float(row.get("duration") or 0)
            if duration <= 0:
                raise ValueError(f"impd at {start} has no duration")
            spans.append((start, start + duration))
    return spans


def _fail_if_mystery_impedance(rows):
    names = unique_event_types(rows)
    if "impd" in names:
        return
    for name in names:
        lower = name.lower()
        if "imp" in lower:
            raise ValueError(
                "no eventType impd, but we got something that looks like impedance: "
                + ", ".join(names)
            )
