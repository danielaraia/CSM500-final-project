import json
import math
import os
import shutil
import tempfile

import numpy as np
import pytest

from alarms import (
    choose_operating_threshold,
    lookback_seconds,
    pick_threshold,
    raise_alarms,
    save_window_scores,
    score_at_threshold,
    silent_seconds,
    test_leads_from_seizures as leads_from_seizures,
    vote_need,
)
from calibrate import (
    discover_runs,
    evaluate_test,
    load_run_context,
    oracle_diagnostic,
    pick_threshold_fast,
    pick_with_fallback,
    run_events_path,
    scorer_for,
)
from calibrate_rules import (
    RULES,
    budget_exceedance_rate,
    empirical_quantile,
    latch_scores,
    rule_hysteresis,
    rule_quantile,
    rule_smooth,
    rule_val_015,
    smooth_scores,
    vote_alarm_probability,
)
from config import write_config
from fast_score import fast_alarms, prepare_scorer, score_part_fast, vote_supremum
from labels import SOP_SECONDS, label_second, LABEL_INTERICTAL
from report_calibrate import fa_ok, parse_args, rule_summaries

SOP_15 = 15 * 60
SOP_5 = 5 * 60


def test_smooth_scores_trailing_mean():
    smoothed = smooth_scores({10: 1.0, 11: 3.0, 12: 5.0}, 2)
    assert smoothed == {10: 1.0, 11: 2.0, 12: 4.0}


def test_smooth_scores_keeps_lone_seconds_after_gaps():
    smoothed = smooth_scores({0: 1.0, 100: 3.0, 101: 7.0}, 10)
    assert smoothed == {0: 1.0, 100: 3.0, 101: 5.0}
    assert set(smoothed) == {0, 100, 101}


def test_smooth_scores_rejects_bad_window():
    with pytest.raises(ValueError, match="at least 1 sec"):
        smooth_scores({0: 1.0}, 0)


def test_latch_scores_enter_exit_hold():
    latched = latch_scores({0: 0.5, 1: 1.5, 2: 2.5, 3: 1.5, 4: 0.7}, 2.0, 1.0)
    assert latched == {0: 0.0, 1: 0.0, 2: 1.0, 3: 1.0, 4: 0.0}


def test_latch_scores_requires_high_above_low():
    with pytest.raises(ValueError, match="below low"):
        latch_scores({0: 1.0}, 1.0, 2.0)


def test_vote_alarm_probability_bounds_and_monotonic():
    assert vote_alarm_probability(0.0, 300, 240) == 0.0
    assert vote_alarm_probability(1.0, 300, 240) == 1.0
    assert abs(vote_alarm_probability(0.3, 50, 1) - (1.0 - 0.7 ** 50)) < 1e-12  # need 1 of n is exact: 1 - (1 - r) ** n
    rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    probs = [vote_alarm_probability(r, 300, 240) for r in rates]
    index = 1
    while index < len(probs):
        assert probs[index] > probs[index - 1]
        index += 1


def test_budget_exceedance_rate_inverts_the_vote_probability():
    for lookback, need in ((300, 240), (150, 120), (50, 40)):
        rate = budget_exceedance_rate(lookback, need, 0.15)
        assert abs(vote_alarm_probability(rate, lookback, need) - 0.15 / 3600.0) < 1e-8
        assert 0.0 < rate < 1.0


def test_empirical_quantile():
    assert empirical_quantile([1.0, 2.0, 3.0, 4.0], 0.25) == 1.0
    assert empirical_quantile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.0
    assert empirical_quantile([1.0, 2.0, 3.0, 4.0], 0.26) == 2.0
    assert empirical_quantile([1.0, 2.0, 3.0, 4.0], 1.0) == 4.0
    with pytest.raises(ValueError, match="no values"):
        empirical_quantile([], 0.5)


def _fuzz_case(rng):
    duration = int(rng.integers(1_500, 4_000))
    seizures = []
    onset = float(rng.integers(0, duration // 4))
    while len(seizures) < rng.integers(1, 4):
        offset = onset + float(rng.integers(5, 60))
        seizures.append((onset, offset))
        onset = offset + float(rng.integers(400, 1_200))
        if onset > duration - 100:
            break
    score_at = {}
    second = 0
    while second < duration:
        if rng.random() < 0.8:
            score_at[second] = float(rng.normal())
        second += 1
    part_start = int(rng.integers(0, duration // 2))
    part_end = part_start + int(rng.integers(1, duration - part_start))
    leads = []
    for onset, _offset in seizures:
        if rng.random() < 0.7:
            leads.append(onset)
    return score_at, seizures, duration, part_start, part_end, leads


def test_fast_scorer_matches_score_at_threshold():
    rng = np.random.default_rng(20260921)
    case = 0
    while case < 25:
        score_at, seizures, duration, part_start, part_end, leads = _fuzz_case(rng)
        for sop_seconds in (SOP_SECONDS, SOP_5):
            scorer = prepare_scorer(score_at, seizures, duration, sop_seconds)
            values = list(score_at.values())
            thresholds = [math.inf, -10.0, 0.0, 0.5]
            if values:
                thresholds.append(float(np.quantile(values, 0.9)))
            for threshold in thresholds:
                expected = score_at_threshold(
                    score_at,
                    threshold,
                    seizures,
                    leads,
                    duration,
                    part_start,
                    part_end,
                    sop_seconds=sop_seconds,
                )
                fast = score_part_fast(
                    scorer, threshold, part_start, part_end, leads
                )
                for key in expected:
                    assert expected[key] == fast[key], (key, threshold, case, sop_seconds)
                expected_alarms = raise_alarms(
                    score_at,
                    threshold,
                    duration,
                    lookback_seconds(sop_seconds),
                    vote_need(lookback_seconds(sop_seconds)),
                    silent_seconds(300, sop_seconds),
                )
                assert expected_alarms == fast["alarms"], (threshold, case, sop_seconds)
        case += 1


def test_fast_scorer_interictal_matches_label_second():
    rng = np.random.default_rng(7)
    _score_at, seizures, duration, _ps, _pe, _leads = _fuzz_case(rng)
    scorer = prepare_scorer({}, seizures, duration, SOP_SECONDS)
    checked = 0
    while checked < 300:
        t = int(rng.integers(0, duration))
        expected = label_second(t, seizures, SOP_SECONDS) == LABEL_INTERICTAL
        assert bool(scorer["interictal"][t]) == expected
        checked += 1


def _synthetic_scores():
    # flat at 0 with two bumps: a long one that can vote an alarm (311 > 240 of 300) and a short one that cannot (51 < 240)
    score_at = {}
    for second in range(0, 20_000):
        score_at[second] = 0.0
    for second in range(8_000, 8_311):
        score_at[second] = 1.0
    for second in range(10_000, 10_051):
        score_at[second] = 0.5
    return score_at


def test_vote_supremum_blocks_all_selection_false_alarms():
    score_at = _synthetic_scores()
    seizures = [(18_000.0, 18_100.0)]
    scorer = prepare_scorer(score_at, seizures, 20_000, SOP_SECONDS)
    supremum = vote_supremum(scorer, 5_000, 15_000)
    # the long 1.0 bump fills 240 of 300 lookback seconds, so the supremum is the bump height. at it no interictal second can vote, below it one can
    assert supremum == 1.0
    at = score_part_fast(scorer, supremum, 5_000, 15_000, [])
    assert at["n_false_alarms"] == 0
    below = score_part_fast(scorer, supremum - 0.5, 5_000, 15_000, [])
    assert below["n_false_alarms"] == 1


def test_vote_supremum_none_without_scores():
    scorer = prepare_scorer({}, [(18_000.0, 18_100.0)], 20_000, SOP_SECONDS)
    assert vote_supremum(scorer, 5_000, 15_000) is None


def test_pick_threshold_fast_matches_pick_threshold():
    score_at = _synthetic_scores()
    seizures = [(18_000.0, 18_100.0)]
    part_start, part_end = 5_000, 15_000
    expected_threshold, expected_info = pick_threshold(
        score_at,
        seizures,
        20_000,
        part_start,
        part_end,
        sop_seconds=SOP_SECONDS,
        fp_max_per_hour=0.15,
    )
    scorer = prepare_scorer(score_at, seizures, 20_000, SOP_SECONDS)
    fast_threshold, fast_info = pick_threshold_fast(
        score_at, seizures, part_start, part_end, SOP_SECONDS, 0.15, scorer=scorer
    )
    assert expected_threshold == fast_threshold
    assert (
        expected_info["val_false_alarms_per_hour"]
        == fast_info["select_false_alarms_per_hour"]
    )
    assert (
        expected_info["val_scorable_interictal_hours"]
        == fast_info["select_scorable_interictal_hours"]
    )


def test_pick_threshold_fast_empty_part_is_inf():
    scorer = prepare_scorer({}, [(18_000.0, 18_100.0)], 20_000, SOP_SECONDS)
    threshold, info = pick_threshold_fast(
        {}, [(18_000.0, 18_100.0)], 5_000, 15_000, SOP_SECONDS, 0.15, scorer=scorer
    )
    assert math.isinf(threshold)
    assert info["select_scorable_interictal_hours"] == 0


def _ctx_for_fallback(score_at, monitor_start, monitor_end):
    return {
        "seizures": [(30_000.0, 30_100.0)],
        "duration_seconds": 40_000,
        "t1": 10_000,
        "t2": 20_000,
        "monitor_start": monitor_start,
        "monitor_end": monitor_end,
        "sop_seconds": SOP_SECONDS,
    }


def test_pick_with_fallback_prefers_val():
    score_at = {}
    for second in range(0, 40_000):
        score_at[second] = 0.0
    ctx = _ctx_for_fallback(score_at, 2_000, 8_000)
    threshold, select_set, _info = pick_with_fallback(ctx, score_at, 0.15)
    assert select_set == "val"
    assert not math.isinf(threshold)


def test_pick_with_fallback_uses_monitor_when_val_is_empty():
    score_at = {}
    for second in range(2_000, 8_000):
        score_at[second] = 0.0
    ctx = _ctx_for_fallback(score_at, 2_000, 8_000)
    threshold, select_set, _info = pick_with_fallback(ctx, score_at, 0.15)
    assert select_set == "monitor"
    assert not math.isinf(threshold)


def test_pick_with_fallback_none_when_both_empty():
    ctx = _ctx_for_fallback({}, None, None)
    threshold, select_set, _info = pick_with_fallback(ctx, {}, 0.15)
    assert math.isinf(threshold)
    assert select_set == "none"


def _write_events(path, seizures, duration_seconds):
    lines = ["onset\tduration\teventType\trecordingDuration"]
    for onset, offset in seizures:
        lines.append(f"{onset}\t{offset - onset}\tsz_focal\t{duration_seconds}")
    lines.append(f"100\t5\tbkgd\t{duration_seconds}")
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def _make_run_tree(root, catch_bump=2.0, fa_bump=1.0):
    seizures = [(10_000.0, 10_060.0), (20_000.0, 20_060.0), (30_000.0, 30_060.0)]
    duration_seconds = 40_000
    t1, t2, test_leads, _n_merged, n_test = leads_from_seizures(
        seizures, SOP_SECONDS
    )
    patient_dir = os.path.join(
        root, "experiments", "runs", "within_patient", "sub-004", "pearson", "sop30",
        "seed123", "eval1",
    )
    os.makedirs(patient_dir)
    write_config(os.path.join(patient_dir, "config.json"), {"sop_minutes": 30})
    val_start = int(math.ceil(t1))
    test_start = int(math.ceil(t2))
    monitor_start = val_start - 5_000
    monitor_end = val_start - 2_000
    val_ends = list(range(val_start, test_start))
    test_ends = list(range(test_start, duration_seconds))
    monitor_ends = list(range(monitor_start, monitor_end))
    val_logits = [0.0] * len(val_ends)
    test_logits = [0.0] * len(test_ends)
    monitor_logits = [0.0] * len(monitor_ends)
    # long high bump late in val, one false alarm at low thresholds
    index = 1_000
    while index < 1_311:
        val_logits[index] = 1.0
        index += 1
    # Long high bump inside the test hit interval of the 30000 seizure. a catch
    hit_second = 29_000 - test_start
    index = hit_second
    while index < hit_second + 311:
        test_logits[index] = catch_bump
        index += 1
    # and a long medium bump in test interictal, a false alarm at low thresholds
    fa_second = 35_000 - test_start
    index = fa_second
    while index < fa_second + 311:
        test_logits[index] = fa_bump
        index += 1
    save_window_scores(os.path.join(patient_dir, "val_scores.npz"), val_ends, val_logits)
    save_window_scores(os.path.join(patient_dir, "test_scores.npz"), test_ends, test_logits)
    save_window_scores(
        os.path.join(patient_dir, "monitor_scores.npz"), monitor_ends, monitor_logits
    )
    events_dir = os.path.join(
        root, "data", "seizeit2", "sub-004", "ses-01", "eeg"
    )
    os.makedirs(events_dir)
    events_path = os.path.join(
        events_dir, "sub-004_ses-01_task-szMonitoring_run-01_events.tsv"
    )
    _write_events(events_path, seizures, duration_seconds)
    metrics = {
        "patient_id": "sub-004",
        "packed_run": "01",
        "t1": t1,
        "t2": t2,
        "n_test_seizures": n_test,
    }
    with open(os.path.join(patient_dir, "metrics.json"), "w") as handle:
        json.dump(metrics, handle)
    return {
        "patient_id": "sub-004",
        "graph_strategy": "pearson",
        "sop_minutes": 30,
        "folder": patient_dir,
    }


def test_discover_runs_and_context_match_the_original_pipeline():
    root = tempfile.mkdtemp()
    try:
        _make_run_tree(root)
        found = discover_runs(root)
        assert len(found) == 1
        assert found[0]["patient_id"] == "sub-004"
        ctx = load_run_context(root, found[0])
        assert ctx["t1"] >= 20_060  # val starts at or after the last train seizure
        assert ctx["t2"] == 25_030.0
        assert ctx["test_leads"] == [30_000.0]
        assert ctx["monitor_start"] == int(math.ceil(ctx["t1"])) - 5_000
        # the baseline rule here has to match choose_operating_threshold plus score_at_threshold exactly
        expected_threshold, expected_info = choose_operating_threshold(
            ctx["score_at"],
            ctx["seizures"],
            ctx["duration_seconds"],
            ctx["t1"],
            ctx["t2"],
            fallback_start=ctx["monitor_start"],
            fallback_end=ctx["monitor_end"],
            sop_seconds=SOP_SECONDS,
        )
        result = rule_val_015(ctx)
        assert result["select_set"] == expected_info["threshold_set"]
        if math.isinf(expected_threshold):
            assert math.isinf(result["threshold"])
        else:
            assert result["threshold"] == expected_threshold
        expected_test = score_at_threshold(
            ctx["score_at"],
            expected_threshold,
            ctx["seizures"],
            ctx["test_leads"],
            ctx["duration_seconds"],
            ctx["t2"],
            ctx["duration_seconds"],
            sop_seconds=SOP_SECONDS,
        )
        mine_test = evaluate_test(ctx, result["eval_map"], result["threshold"])
        assert mine_test["n_false_alarms"] == expected_test["n_false_alarms"]
        assert mine_test["n_seizures_caught"] == expected_test["n_seizures_caught"]
        assert mine_test["n_seizures_caught"] == 1  # the 2.0 bump catches at any budget
        assert mine_test["n_false_alarms"] == 1  # the 1.0 bump is a false alarm
        assert (
            mine_test["scorable_interictal_hours"]
            == expected_test["scorable_interictal_hours"]
        )
    finally:
        shutil.rmtree(root)


def test_all_rules_run_on_a_synthetic_context():
    root = tempfile.mkdtemp()
    try:
        _make_run_tree(root)
        ctx = load_run_context(root, discover_runs(root)[0])
        for name, rule_fn in RULES:
            result = rule_fn(ctx)
            assert result["select_set"] in ("val", "monitor", "pool", "none")
            assert set(result["eval_map"]) == set(ctx["score_at"])
            test_scored = evaluate_test(ctx, result["eval_map"], result["threshold"])
            assert test_scored["n_seizures_evaluable"] == 1
        # hysteresis and smooth hand back changed maps
        hysteresis = rule_hysteresis(ctx)
        assert set(hysteresis["eval_map"].values()) <= {0.0, 1.0}
        smoothed = rule_smooth(ctx)
        assert set(smoothed["eval_map"]) == set(ctx["score_at"])
        quantile = rule_quantile(ctx)
        assert not math.isinf(quantile["threshold"])
    finally:
        shutil.rmtree(root)


def test_oracle_diagnostic_separable_when_catch_is_above_the_fa_bump():
    root = tempfile.mkdtemp()
    try:
        _make_run_tree(root, catch_bump=2.0, fa_bump=1.0)
        ctx = load_run_context(root, discover_runs(root)[0])
        out = oracle_diagnostic(ctx, 1)
        assert out["test_vote_supremum"] == 1.0
        assert out["oracle_separable"] is True
        assert out["oracle_n_false_at_catch"] == 0
        assert out["oracle_caught_at_zero_fa"] == 1
    finally:
        shutil.rmtree(root)


def test_oracle_diagnostic_inseparable_when_fa_bump_is_above_the_catch():
    root = tempfile.mkdtemp()
    try:
        _make_run_tree(root, catch_bump=1.0, fa_bump=2.0)
        ctx = load_run_context(root, discover_runs(root)[0])
        out = oracle_diagnostic(ctx, 1)
        # the interictal bump votes up to 2.0, the catch needs 1.0, where the bump false alarms
        assert out["test_vote_supremum"] == 2.0
        assert out["oracle_separable"] is False
        assert out["oracle_n_false_at_catch"] == 1
    finally:
        shutil.rmtree(root)


def test_run_events_path_requires_one_file(tmp_path):
    with pytest.raises(ValueError, match="exactly one local events file"):
        run_events_path(str(tmp_path), "sub-999", None)


def test_parse_args():
    assert parse_args(["report_calibrate.py"]) is None
    assert parse_args(["report_calibrate.py", "sub-051", "mi", "15"]) == [
        ("sub-051", "mi", 15)
    ]
    with pytest.raises(ValueError, match="usage"):
        parse_args(["report_calibrate.py", "sub-051"])


def test_fa_ok_follows_the_study_rule():
    assert fa_ok({"scorable_interictal_hours": 10.0, "false_alarms_per_hour": 0.1, "n_false_alarms": 1})
    assert not fa_ok({"scorable_interictal_hours": 10.0, "false_alarms_per_hour": 0.2, "n_false_alarms": 2})
    assert fa_ok({"scorable_interictal_hours": 1.0, "false_alarms_per_hour": 0.0, "n_false_alarms": 0})
    assert not fa_ok({"scorable_interictal_hours": 1.0, "false_alarms_per_hour": 1.0, "n_false_alarms": 1})


def test_rule_summaries_counts():
    gap_rows = [{"n_seizures_caught": 1}, {"n_seizures_caught": 0}]
    rule_rows = [
        {
            "rule": "val_0_15",
            "patient_id": "sub-004",
            "graph_strategy": "pearson",
            "sop_minutes": 30,
            "fa_ok": True,
            "catches_kept": True,
            "study_criterion_met": True,
            "n_seizures_caught": 1,
            "false_alarms_per_hour": 0.0,
        },
        {
            "rule": "val_0_15",
            "patient_id": "sub-051",
            "graph_strategy": "mi",
            "sop_minutes": 15,
            "fa_ok": False,
            "catches_kept": False,
            "study_criterion_met": False,
            "n_seizures_caught": 0,
            "false_alarms_per_hour": 2.5,
        },
    ]
    summaries = rule_summaries(rule_rows, gap_rows)
    assert len(summaries) == len(RULES)
    first = summaries[0]
    assert first["rule"] == "val_0_15"
    assert first["n_fa_ok"] == 1
    assert first["caught_total"] == 1
    assert first["baseline_caught_total"] == 1
    assert first["worst_false_alarms_per_hour"] == 2.5
