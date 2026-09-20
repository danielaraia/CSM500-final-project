import glob
import math
import os

from alarms import (
    load_window_scores,
    merge_lead_onsets,
    score_map,
    silent_seconds,
    test_leads_from_seizures,
)
from events import load_event_rows, seizure_spans
from fast_score import prepare_scorer, score_part_fast, vote_supremum
from labels import SPH_SECONDS, seizures_with_full_sop
from metrics import read_metrics
from config import read_config
from score_alarms import recording_duration_seconds


def discover_runs(here):
    root = os.path.join(here, "experiments", "runs", "within_patient")
    runs = []
    if not os.path.isdir(root):
        return runs
    for patient_id in sorted(os.listdir(root)):
        patient_dir = os.path.join(root, patient_id)
        if not os.path.isdir(patient_dir):
            continue
        for graph_strategy in sorted(os.listdir(patient_dir)):
            graph_dir = os.path.join(patient_dir, graph_strategy)
            if not os.path.isdir(graph_dir):
                continue
            for sop_name in sorted(os.listdir(graph_dir)):
                if not sop_name.startswith("sop"):
                    continue
                sop_minutes = int(sop_name[3:])
                folder = os.path.join(graph_dir, sop_name, "seed123", "eval1")
                if not os.path.isfile(os.path.join(folder, "test_scores.npz")):
                    continue
                if not os.path.isfile(os.path.join(folder, "val_scores.npz")):
                    continue
                if not os.path.isfile(os.path.join(folder, "metrics.json")):
                    continue
                runs.append(
                    {
                        "patient_id": patient_id,
                        "graph_strategy": graph_strategy,
                        "sop_minutes": sop_minutes,
                        "folder": folder,
                    }
                )
    return runs


def run_events_path(here, patient_id, packed_run):
    base = os.path.join(here, "data", "seizeit2", patient_id, "ses-01", "eeg")
    if packed_run is not None and str(packed_run).strip() != "":
        path = os.path.join(
            base, f"{patient_id}_ses-01_task-szMonitoring_run-{packed_run}_events.tsv"
        )
        if not os.path.isfile(path):
            raise ValueError(f"events file is missing: {path}")
        return path
    # fallback: the oldest runs never saved packed_run, so use the one local file
    matches = glob.glob(
        os.path.join(base, f"{patient_id}_ses-01_task-szMonitoring_run-*_events.tsv")
    )
    if len(matches) != 1:
        raise ValueError(
            f"need exactly one local events file for {patient_id}, we got {len(matches)}"
        )
    return matches[0]


def build_score_map(folder):
    val_ends, val_logits = load_window_scores(os.path.join(folder, "val_scores.npz"))
    test_ends, test_logits = load_window_scores(os.path.join(folder, "test_scores.npz"))
    score_at = score_map(val_ends, val_logits)
    for second, value in score_map(test_ends, test_logits).items():
        if second in score_at:
            raise ValueError(f"val and test both scored second {second}")
        score_at[second] = value
    monitor_path = os.path.join(folder, "monitor_scores.npz")
    monitor_start = None
    monitor_end = None
    n_monitor = 0
    if os.path.isfile(monitor_path):
        monitor_ends, monitor_logits = load_window_scores(monitor_path)
        monitor_map = score_map(monitor_ends, monitor_logits)
        for second, value in monitor_map.items():
            if second in score_at:
                raise ValueError(f"monitor and val or test both scored second {second}")
            score_at[second] = value
        if len(monitor_ends) > 0:
            monitor_start = int(min(monitor_ends))
            monitor_end = int(max(monitor_ends)) + 1
        n_monitor = int(len(monitor_ends))
    return score_at, monitor_start, monitor_end, n_monitor


def load_run_context(here, run):
    folder = run["folder"]
    cfg = read_config(os.path.join(folder, "config.json"))
    sop_minutes = cfg.get("sop_minutes")
    if sop_minutes != run["sop_minutes"]:
        raise ValueError(
            "config sop does not match the folder, "
            f"config {sop_minutes!r} folder {run['sop_minutes']!r} in {folder}"
        )
    metrics = read_metrics(folder)
    events_path = run_events_path(here, run["patient_id"], metrics.get("packed_run"))
    rows = load_event_rows(events_path)
    seizures = seizure_spans(rows)
    duration_seconds = recording_duration_seconds(rows)
    sop_seconds = sop_minutes * 60
    t1, t2, test_leads, n_merged, n_test = test_leads_from_seizures(
        seizures, sop_seconds
    )
    if "t1" in metrics and abs(float(metrics["t1"]) - t1) > 1e-6:
        raise ValueError(
            f"t1 in metrics does not match the events file, "
            f"metrics {metrics['t1']} events {t1} in {folder}"
        )
    if "t2" in metrics and abs(float(metrics["t2"]) - t2) > 1e-6:
        raise ValueError(
            f"t2 in metrics does not match the events file, "
            f"metrics {metrics['t2']} events {t2} in {folder}"
        )
    if "n_test_seizures" in metrics and int(metrics["n_test_seizures"]) != n_test:
        raise ValueError(
            f"n_test_seizures in metrics does not match the events file, "
            f"metrics {metrics['n_test_seizures']} events {n_test} in {folder}"
        )
    score_at, monitor_start, monitor_end, n_monitor = build_score_map(folder)
    usable = seizures_with_full_sop(seizures, sop_seconds)
    onsets = []
    for onset, _offset in usable:
        onsets.append(onset)
    all_leads, _n_merged = merge_lead_onsets(
        onsets, silent_seconds(SPH_SECONDS, sop_seconds)
    )
    return {
        "patient_id": run["patient_id"],
        "graph_strategy": run["graph_strategy"],
        "sop_minutes": sop_minutes,
        "sop_seconds": sop_seconds,
        "folder": folder,
        "events_path": events_path,
        "metrics": metrics,
        "seizures": seizures,
        "duration_seconds": duration_seconds,
        "t1": t1,
        "t2": t2,
        "test_leads": test_leads,
        "all_leads": all_leads,
        "score_at": score_at,
        "monitor_start": monitor_start,
        "monitor_end": monitor_end,
        "n_monitor_windows_with_scores": n_monitor,
    }


def scorer_for(ctx, score_at, probe_cache=None):
    # one fast scorer per score map, cached
    key = ("scorer", id(score_at))  # id() is ok, each map outlives one report run
    if probe_cache is not None and key in probe_cache:
        return probe_cache[key]
    scorer = prepare_scorer(
        score_at, ctx["seizures"], ctx["duration_seconds"], ctx["sop_seconds"]
    )
    if probe_cache is not None:
        probe_cache[key] = scorer
    return scorer


def pick_threshold_fast(
    score_at,
    seizures,
    part_start,
    part_end,
    sop_seconds,
    fp_max_per_hour,
    probe_cache=None,
    scorer=None,
):
    # Same walk as alarms.pick_threshold, but probes use the fast scorer and the
    # walk starts just under the vote supremum. first version was pure python, minutes per threshold
    if scorer is None:
        raise ValueError("pick_threshold_fast needs a scorer from scorer_for")

    def probe(threshold):
        key = (id(score_at), part_start, part_end, threshold)
        if probe_cache is not None and key in probe_cache:
            return probe_cache[key]
        scored = score_part_fast(scorer, threshold, part_start, part_end, [])
        if probe_cache is not None:
            probe_cache[key] = scored
        return scored

    first = probe(math.inf)
    hours = first["scorable_interictal_hours"]
    if hours <= 0:
        return math.inf, {
            "select_false_alarms_per_hour": None,
            "select_scorable_interictal_hours": hours,
        }
    seen = {}
    unique = []
    for second, value in score_at.items():
        if part_start <= second < part_end and value not in seen:
            seen[value] = True
            unique.append(value)
    unique.sort(reverse=True)
    supremum_key = ("vsup", id(score_at), part_start, part_end)
    if probe_cache is not None and supremum_key in probe_cache:
        supremum = probe_cache[supremum_key]
    else:
        supremum = vote_supremum(scorer, part_start, part_end)
        if probe_cache is not None:
            probe_cache[supremum_key] = supremum
    chosen = math.inf
    chosen_fpr = 0.0
    index = 0
    if supremum is not None:
        # all of these score zero false alarms, so chosen ends on the smallest
        while index < len(unique) and unique[index] >= supremum:
            chosen = unique[index]
            chosen_fpr = 0.0
            index += 1
    while index < len(unique):
        threshold = unique[index]
        fpr = probe(threshold)["false_alarms_per_hour"]
        if fpr is None:
            raise ValueError("part fpr is missing even though scorable hours were > 0")
        if fpr <= fp_max_per_hour:
            chosen = threshold
            chosen_fpr = fpr
        else:
            break
        index += 1
    return chosen, {
        "select_false_alarms_per_hour": chosen_fpr,
        "select_scorable_interictal_hours": hours,
    }


def pick_with_fallback(ctx, score_at, fp_max_per_hour, probe_cache=None):
    # same idea as alarms.choose_operating_threshold: val first, monitor when val has nothing, inf when neither picks
    scorer = scorer_for(ctx, score_at, probe_cache=probe_cache)
    threshold, info = pick_threshold_fast(
        score_at,
        ctx["seizures"],
        ctx["t1"],
        ctx["t2"],
        ctx["sop_seconds"],
        fp_max_per_hour,
        probe_cache=probe_cache,
        scorer=scorer,
    )
    if info["select_scorable_interictal_hours"] > 0:
        return threshold, "val", info
    if ctx["monitor_start"] is None:
        return math.inf, "none", info
    threshold, mon_info = pick_threshold_fast(
        score_at,
        ctx["seizures"],
        ctx["monitor_start"],
        ctx["monitor_end"],
        ctx["sop_seconds"],
        fp_max_per_hour,
        probe_cache=probe_cache,
        scorer=scorer,
    )
    if mon_info["select_scorable_interictal_hours"] > 0:
        return threshold, "monitor", mon_info
    return math.inf, "none", mon_info


def selection_part(ctx, select_set):
    if select_set == "val":
        return ctx["t1"], ctx["t2"]
    if select_set == "monitor":
        return ctx["monitor_start"], ctx["monitor_end"]
    if select_set == "pool":
        return ctx["monitor_start"], ctx["t2"]
    raise ValueError(f"no selection part for set {select_set!r}")


def evaluate_test(ctx, eval_map, threshold, probe_cache=None):
    scorer = scorer_for(ctx, eval_map, probe_cache=probe_cache)
    return score_part_fast(
        scorer, threshold, ctx["t2"], ctx["duration_seconds"], ctx["test_leads"]
    )


def evaluate_part(ctx, eval_map, threshold, part_start, part_end, probe_cache=None):
    scorer = scorer_for(ctx, eval_map, probe_cache=probe_cache)
    return score_part_fast(scorer, threshold, part_start, part_end, [])


def oracle_diagnostic(ctx, baseline_caught, probe_cache=None, max_probes=2000):
    # Fitted on test, never used to pick anything. could ANY threshold keep the baseline
    # catches with zero test false alarms. walk down from the test vote supremum until the catches come back
    scorer = scorer_for(ctx, ctx["score_at"], probe_cache=probe_cache)
    t2 = ctx["t2"]
    duration = ctx["duration_seconds"]
    t_zero = vote_supremum(scorer, t2, duration)
    out = {"test_vote_supremum": t_zero}
    if t_zero is None:
        out["oracle_separable"] = True
        out["oracle_note"] = "no test interictal window can vote at any threshold"
        return out
    scored = score_part_fast(scorer, t_zero, t2, duration, ctx["test_leads"])
    out["oracle_caught_at_zero_fa"] = scored["n_seizures_caught"]
    if scored["n_seizures_caught"] >= baseline_caught:
        out["oracle_separable"] = True
        out["oracle_threshold"] = t_zero
        out["oracle_n_false_at_catch"] = scored["n_false_alarms"]
        return out
    seen = {}
    unique = []
    for second, value in ctx["score_at"].items():
        if t2 <= second < duration and value < t_zero and value not in seen:
            seen[value] = True
            unique.append(value)
    unique.sort(reverse=True)
    index = 0
    while index < len(unique) and index < max_probes:
        threshold = unique[index]
        key = (id(ctx["score_at"]), t2, duration, threshold)
        if probe_cache is not None and key in probe_cache:
            scored = probe_cache[key]
        else:
            scored = score_part_fast(scorer, threshold, t2, duration, ctx["test_leads"])
            if probe_cache is not None:
                probe_cache[key] = scored
        if scored["n_seizures_caught"] >= baseline_caught:
            out["oracle_separable"] = scored["n_false_alarms"] == 0
            out["oracle_threshold"] = threshold
            out["oracle_n_false_at_catch"] = scored["n_false_alarms"]
            out["oracle_note"] = "first catch restoring threshold, lower ones unchecked"
            return out
        index += 1
    out["oracle_separable"] = False
    if index >= max_probes and index < len(unique):
        out["oracle_note"] = f"walk capped at {max_probes} probes, catch not restored"
    else:
        out["oracle_note"] = "catch never restored at any test threshold"
    return out
