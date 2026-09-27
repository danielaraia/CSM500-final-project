import json
import math
import os
import sys

from evaluation.alarm_leads import hit_interval
from evaluation.alarms import (
    FP_MAX_PER_HOUR,
    HOURS_TO_RESOLVE_FP_MAX,
    alarm_hits_a_seizure,
    study_criterion_met,
)
from evaluation.calibrate import (
    discover_runs,
    evaluate_part,
    evaluate_test,
    load_run_context,
    oracle_diagnostic,
    selection_part,
)
from evaluation.calibrate_rules import RULES
from data.labels import SPH_SECONDS

SUMMARY_NAME = "results/p5-calibration-summary.json"


def parse_args(argv):
    if len(argv) == 1:
        return None
    if len(argv) == 4:
        return [(argv[1], argv[2], int(argv[3]))]
    raise ValueError(
        "usage: python report_calibrate.py [patient_id graph_strategy sop_minutes]"
    )


def fa_ok(test_scored):
    # the budget is 0.15 per hour
    hours = test_scored["scorable_interictal_hours"]
    if hours >= HOURS_TO_RESOLVE_FP_MAX:
        return (
            test_scored["false_alarms_per_hour"] is not None
            and test_scored["false_alarms_per_hour"] <= FP_MAX_PER_HOUR
        )
    return test_scored["n_false_alarms"] == 0  # under 6.67 h even one FA breaks it, so ask for zero


def selection_metrics(ctx, eval_map, threshold, select_set, probe_cache):
    if select_set == "none":
        return None
    part_start, part_end = selection_part(ctx, select_set)
    return evaluate_part(
        ctx, eval_map, threshold, part_start, part_end, probe_cache=probe_cache
    )


def selection_hits(ctx, alarms, select_set):
    # Hits and crossing leads inside the selection slice. selection slices hold train seizures, so this is catching on the selection side
    if select_set == "none":
        return 0, 0
    part_start, part_end = selection_part(ctx, select_set)
    hits = 0
    for alarm_time in alarms:
        if part_start <= alarm_time < part_end and alarm_hits_a_seizure(
            alarm_time, ctx["all_leads"], SPH_SECONDS, ctx["sop_seconds"]
        ):
            hits += 1
    crossing = 0
    for lead in ctx["all_leads"]:
        start, end = hit_interval(lead, SPH_SECONDS, ctx["sop_seconds"])
        if start < part_end and end > part_start:
            crossing += 1
    return hits, crossing


def evaluate_rule(ctx, rule_name, rule_fn, baseline_caught, probe_cache):
    result = rule_fn(ctx, probe_cache=probe_cache)
    eval_map = result["eval_map"]
    threshold = result["threshold"]
    select_set = result["select_set"]
    test_scored = evaluate_test(ctx, eval_map, threshold, probe_cache=probe_cache)
    sel_scored = selection_metrics(ctx, eval_map, threshold, select_set, probe_cache)
    sel_hits, sel_crossing = selection_hits(ctx, test_scored["alarms"], select_set)
    row = {
        "patient_id": ctx["patient_id"],
        "graph_strategy": ctx["graph_strategy"],
        "sop_minutes": ctx["sop_minutes"],
        "rule": rule_name,
        "note": result["note"],
        "threshold": None if math.isinf(threshold) else float(threshold),
        "select_set": select_set,
        "select_false_alarms_per_hour": (
            None if sel_scored is None else sel_scored["false_alarms_per_hour"]
        ),
        "select_scorable_interictal_hours": (
            None if sel_scored is None else sel_scored["scorable_interictal_hours"]
        ),
        "select_hit_alarms": sel_hits,
        "select_leads_crossing": sel_crossing,
        "n_false_alarms": test_scored["n_false_alarms"],
        "false_alarms_per_hour": test_scored["false_alarms_per_hour"],
        "scorable_interictal_hours": test_scored["scorable_interictal_hours"],
        "resolves_0_15_per_hour": test_scored["resolves_0_15_per_hour"],
        "n_seizures_caught": test_scored["n_seizures_caught"],
        "n_seizures_evaluable": test_scored["n_seizures_evaluable"],
        "event_sensitivity": test_scored["event_sensitivity"],
        "fa_ok": fa_ok(test_scored),
        "catches_kept": test_scored["n_seizures_caught"] >= baseline_caught,
        "study_criterion_met": study_criterion_met(
            test_scored["event_sensitivity"],
            test_scored["false_alarms_per_hour"],
            test_scored["n_false_alarms"],
            test_scored["resolves_0_15_per_hour"],
        ),
    }
    return row


def cross_check_baseline(ctx, row):
    # the baseline row redoes the original pipeline, so it has to match what the run folder recorded at scoring time
    metrics = ctx["metrics"]
    problems = []
    if "alarm_threshold" in metrics:
        recorded = metrics["alarm_threshold"]
        got = row["threshold"]
        if recorded is None:
            if got is not None:
                problems.append(f"threshold recorded None recomputed {got}")
        elif got is None or abs(recorded - got) > 1e-9:
            problems.append(f"threshold recorded {recorded} recomputed {got}")
    pairs = (
        ("false_alarms_per_hour", "false_alarms_per_hour", 1e-6),
        ("n_false_alarms", "n_false_alarms", 0),
        ("n_seizures_caught", "n_seizures_caught", 0),
        ("n_seizures_evaluable", "n_seizures_evaluable", 0),
        ("scorable_interictal_hours", "scorable_interictal_hours", 1e-6),
    )
    for recorded_key, row_key, tol in pairs:
        if recorded_key not in metrics:
            continue
        recorded = metrics[recorded_key]
        got = row[row_key]
        if recorded is None or got is None:
            if recorded != got:
                problems.append(f"{recorded_key} recorded {recorded} recomputed {got}")
        elif abs(recorded - got) > tol:
            problems.append(f"{recorded_key} recorded {recorded} recomputed {got}")
    if "threshold_set" in metrics and row["select_set"] in ("val", "monitor"):
        if metrics["threshold_set"] != row["select_set"]:
            problems.append(
                f"threshold_set recorded {metrics['threshold_set']} "
                f"recomputed {row['select_set']}"
            )
    return problems


def run_key(row):
    return (row["patient_id"], row["graph_strategy"], row["sop_minutes"])


def fmt_fa(fa_per_hour, n_false):
    if fa_per_hour is None:
        return f"- ({n_false})"
    return f"{fa_per_hour:.3f} ({n_false})"


def gap_table(gap_rows):
    head = (
        "patient | graph | sop | set | threshold | sel fa/h | sel hits | "
        "test fa/h (n) | caught | evaluable"
    )
    lines = [head]
    for row in gap_rows:
        threshold = row["threshold"]
        threshold_text = "inf" if threshold is None else f"{threshold:.3f}"
        sel_fa = row["select_false_alarms_per_hour"]
        sel_fa_text = "-" if sel_fa is None else f"{sel_fa:.3f}"
        lines.append(
            f"{row['patient_id']} | {row['graph_strategy']} | {row['sop_minutes']} | "
            f"{row['select_set']} | {threshold_text} | {sel_fa_text} | "
            f"{row['select_hit_alarms']}/{row['select_leads_crossing']} | "
            f"{fmt_fa(row['false_alarms_per_hour'], row['n_false_alarms'])} | "
            f"{row['n_seizures_caught']} | {row['n_seizures_evaluable']}"
        )
    return "\n".join(lines)


def oracle_table(gap_rows):
    head = (
        "patient | graph | sop | baseline thr | sel supremum | test supremum | "
        "caught at zero fa | oracle separable"
    )
    lines = [head]
    for row in gap_rows:
        baseline = row["threshold"]
        baseline_text = "inf" if baseline is None else f"{baseline:.3f}"
        sel_sup = row.get("vote_supremum_threshold")
        sel_sup_text = "inf" if sel_sup is None else f"{sel_sup:.3f}"
        test_sup = row.get("test_vote_supremum")
        test_sup_text = "-" if test_sup is None else f"{test_sup:.3f}"
        caught_zero = row.get("oracle_caught_at_zero_fa", 0)
        lines.append(
            f"{row['patient_id']} | {row['graph_strategy']} | {row['sop_minutes']} | "
            f"{baseline_text} | {sel_sup_text} | {test_sup_text} | "
            f"{caught_zero}/{row['n_seizures_caught']} | {row.get('oracle_separable')}"
        )
    return "\n".join(lines)


def rule_matrix(rule_rows):
    rule_names = []
    for name, _fn in RULES:
        rule_names.append(name)
    by_run = {}
    for row in rule_rows:
        by_run.setdefault(run_key(row), {})[row["rule"]] = row
    head = "patient | graph | sop | " + " | ".join(rule_names)
    lines = [head]
    for key in sorted(by_run):
        cells = []
        for name in rule_names:
            row = by_run[key][name]
            cells.append(
                f"{fmt_fa(row['false_alarms_per_hour'], row['n_false_alarms'])} "
                f"c{row['n_seizures_caught']}"
            )
        lines.append(" | ".join(str(part) for part in key) + " | " + " | ".join(cells))
    return "\n".join(lines)


def rule_summaries(rule_rows, gap_rows):
    baseline_caught_total = 0
    for row in gap_rows:
        baseline_caught_total += row["n_seizures_caught"]
    summaries = []
    for name, _fn in RULES:
        rows = [row for row in rule_rows if row["rule"] == name]
        n_fa_ok = 0
        n_catches_kept = 0
        n_study_met = 0
        caught_total = 0
        worst_fa = 0.0
        for row in rows:
            if row["fa_ok"]:
                n_fa_ok += 1
            if row["catches_kept"]:
                n_catches_kept += 1
            if row["study_criterion_met"]:
                n_study_met += 1
            caught_total += row["n_seizures_caught"]
            if row["false_alarms_per_hour"] is not None:
                worst_fa = max(worst_fa, row["false_alarms_per_hour"])
        summaries.append(
            {
                "rule": name,
                "n_runs": len(rows),
                "n_fa_ok": n_fa_ok,
                "n_catches_kept": n_catches_kept,
                "n_study_criterion_met": n_study_met,
                "caught_total": caught_total,
                "baseline_caught_total": baseline_caught_total,
                "worst_false_alarms_per_hour": worst_fa,
            }
        )
    return summaries


def summary_table(summaries):
    lines = [
        "rule | runs | fa ok | catches kept | study met | caught | baseline caught | worst fa/h"
    ]
    for summary in summaries:
        lines.append(
            f"{summary['rule']} | {summary['n_runs']} | {summary['n_fa_ok']} | "
            f"{summary['n_catches_kept']} | {summary['n_study_criterion_met']} | "
            f"{summary['caught_total']} | {summary['baseline_caught_total']} | "
            f"{summary['worst_false_alarms_per_hour']:.3f}"
        )
    return "\n".join(lines)


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    only = parse_args(sys.argv)
    runs = discover_runs(here)
    if only is not None:
        wanted = set(only)
        runs = [
            run
            for run in runs
            if (run["patient_id"], run["graph_strategy"], run["sop_minutes"]) in wanted
        ]
        if len(runs) == 0:
            raise ValueError(f"no scored run matches {only}")
    gap_rows = []
    rule_rows = []
    problems = []
    for run in runs:
        ctx = load_run_context(here, run)
        baseline_caught = ctx["metrics"].get("n_seizures_caught", 0)
        probe_cache = {}
        rows_for_run = {}
        for rule_name, rule_fn in RULES:
            row = evaluate_rule(ctx, rule_name, rule_fn, baseline_caught, probe_cache)
            rows_for_run[rule_name] = row
            rule_rows.append(row)
        gap_row = dict(rows_for_run["val_0_15"])
        gap_row["vote_supremum_threshold"] = rows_for_run["vote_supremum"]["threshold"]
        for key, value in oracle_diagnostic(ctx, baseline_caught, probe_cache).items():
            gap_row[key] = value
        gap_rows.append(gap_row)
        for problem in cross_check_baseline(ctx, gap_row):
            problems.append(f"{run['patient_id']} {run['graph_strategy']} {run['sop_minutes']}: {problem}")
        print(
            f"done {run['patient_id']} {run['graph_strategy']} sop{run['sop_minutes']}"
        )
    summaries = rule_summaries(rule_rows, gap_rows)
    summary = {
        "command": "python report_calibrate.py",
        "n_runs": len(runs),
        "gap_rows": gap_rows,
        "rule_rows": rule_rows,
        "rule_summaries": summaries,
        "cross_check_problems": problems,
    }
    out = os.path.join(here, SUMMARY_NAME)
    with open(out, "w") as handle:
        json.dump(summary, indent=2, fp=handle)
        handle.write("\n")
    print()
    print("gap table (baseline rule val_0_15):")
    print(gap_table(gap_rows))
    print()
    print("oracle diagnostic (test fitted, never used for selection):")
    print(oracle_table(gap_rows))
    print()
    print("rule matrix, test fa/h (n) and catches:")
    print(rule_matrix(rule_rows))
    print()
    print(summary_table(summaries))
    print()
    if problems:
        print("CROSS CHECK PROBLEMS:")
        for problem in problems:
            print(" ", problem)
    else:
        print("cross check: baseline recomputation matches every metrics.json")
    print(f"evaluated {len(RULES)} rules on {len(runs)} runs")


if __name__ == "__main__":
    main()
