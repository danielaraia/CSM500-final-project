import math

from alarms import FP_MAX_PER_HOUR, lookback_seconds, vote_need
from calibrate import (
    evaluate_part,
    pick_threshold_fast,
    pick_with_fallback,
    scorer_for,
)
from fast_score import vote_supremum
from labels import LABEL_INTERICTAL, label_second

SAFETY_FP_MAX_PER_HOUR = 0.05
HYSTERESIS_LOW_FP_MAX_PER_HOUR = 1.5
SECONDS_PER_HOUR = 3600.0


def smooth_scores(score_at, window_seconds):
    # Trailing mean over the vote window. seconds whose window holds only their own
    # score keep it, so scorable hours do not move
    if window_seconds < 1:
        raise ValueError(f"window has to be at least 1 sec, we got {window_seconds}")
    smoothed = {}
    seconds = sorted(score_at)
    running = 0.0
    n_in = 0
    tail = 0
    for second in seconds:
        running += score_at[second]
        n_in += 1
        while seconds[tail] <= second - window_seconds:
            running -= score_at[seconds[tail]]
            n_in -= 1
            tail += 1
        smoothed[second] = running / n_in
    return smoothed


def latch_scores(score_at, high, low):
    # on above high, off below low, hold in between. only scored seconds get a state, so the counts we divide by stay the same
    if high < low:
        raise ValueError(f"high threshold is below low, we got {high} {low}")
    latched = {}
    state = False
    for second in sorted(score_at):
        value = score_at[second]
        if value > high:
            state = True
        elif value < low:
            state = False
        latched[second] = 1.0 if state else 0.0
    return latched


def vote_alarm_probability(rate, lookback, need):
    # P(an alarm votes at one second) = P(Binomial(lookback, rate) >= need), if each second sits above threshold on its own with probability rate
    if rate < 0.0 or rate > 1.0:
        raise ValueError(f"rate has to be in [0, 1], we got {rate}")
    total = 0.0
    k = need
    while k <= lookback:
        total += (
            float(math.comb(lookback, k))
            * (rate ** k)
            * ((1.0 - rate) ** (lookback - k))
        )
        k += 1
    return total


def budget_exceedance_rate(lookback, need, fp_max_per_hour):
    # the above threshold rate that gives fp_max alarms per hour when scores are independent. set by the budget, not by data
    target = fp_max_per_hour / SECONDS_PER_HOUR  # solve vote_alarm_probability == this by bisection
    lo = 0.0
    hi = 1.0
    index = 0
    while index < 80:
        mid = (lo + hi) / 2.0
        if vote_alarm_probability(mid, lookback, need) < target:
            lo = mid
        else:
            hi = mid
        index += 1
    return (lo + hi) / 2.0


def empirical_quantile(values, level):
    # the smallest value with at least level fraction of the values at or below it
    if len(values) == 0:
        raise ValueError("cant take a quantile of no values")
    if level <= 0.0:
        return min(values)
    if level >= 1.0:
        return max(values)
    ordered = sorted(values)
    index = int(math.ceil(level * len(ordered))) - 1
    if index < 0:
        index = 0
    return ordered[index]


def selection_interictal_values(ctx, part_start, part_end):
    values = []
    second = int(math.ceil(part_start))
    while second < part_end:
        if second in ctx["score_at"]:
            if label_second(second, ctx["seizures"], ctx["sop_seconds"]) == LABEL_INTERICTAL:
                values.append(ctx["score_at"][second])
        second += 1
    return values


def _rule_result(eval_map, threshold, select_set, note):
    return {
        "eval_map": eval_map,
        "threshold": threshold,
        "select_set": select_set,
        "note": note,
    }


def _budget_rule(ctx, fp_max_per_hour, note, probe_cache=None):
    threshold, select_set, _info = pick_with_fallback(
        ctx, ctx["score_at"], fp_max_per_hour, probe_cache=probe_cache
    )
    return _rule_result(ctx["score_at"], threshold, select_set, note)


def rule_val_015(ctx, probe_cache=None):
    return _budget_rule(ctx, FP_MAX_PER_HOUR, "locked rule: val budget 0.15", probe_cache)


def rule_val_005(ctx, probe_cache=None):
    return _budget_rule(
        ctx, SAFETY_FP_MAX_PER_HOUR, "safety margin: val budget 0.05", probe_cache
    )


def rule_vote_supremum(ctx, probe_cache=None):
    # Zero selection false alarms, built in: the threshold is the vote supremum (see fast_score).
    # note: first tried as the original walk at budget 0. this is stronger and cheaper
    scorer = scorer_for(ctx, ctx["score_at"], probe_cache=probe_cache)
    select_set = "none"
    part = None
    val_probe = evaluate_part(
        ctx, ctx["score_at"], math.inf, ctx["t1"], ctx["t2"], probe_cache=probe_cache
    )
    if val_probe["scorable_interictal_hours"] > 0:
        select_set = "val"
        part = (ctx["t1"], ctx["t2"])
    elif ctx["monitor_start"] is not None:
        mon_probe = evaluate_part(
            ctx,
            ctx["score_at"],
            math.inf,
            ctx["monitor_start"],
            ctx["monitor_end"],
            probe_cache=probe_cache,
        )
        if mon_probe["scorable_interictal_hours"] > 0:
            select_set = "monitor"
            part = (ctx["monitor_start"], ctx["monitor_end"])
    if part is None:
        return _rule_result(ctx["score_at"], math.inf, "none", "no selection data")
    threshold = vote_supremum(scorer, part[0], part[1])
    if threshold is None:
        return _rule_result(
            ctx["score_at"], math.inf, select_set, "no interictal votes in selection"
        )
    return _rule_result(
        ctx["score_at"],
        threshold,
        select_set,
        "vote supremum: zero selection FA by construction",
    )


def rule_pool_015(ctx, probe_cache=None):
    # more selection data: monitor start to val end at 0.15. falls back to plain val 0.15 when there are no monitor scores or the pool has no scorable interictal time
    if ctx["monitor_start"] is None:
        return rule_val_015(ctx, probe_cache)
    threshold, info = pick_threshold_fast(
        ctx["score_at"],
        ctx["seizures"],
        ctx["monitor_start"],
        ctx["t2"],
        ctx["sop_seconds"],
        FP_MAX_PER_HOUR,
        probe_cache=probe_cache,
        scorer=scorer_for(ctx, ctx["score_at"], probe_cache=probe_cache),
    )
    if info["select_scorable_interictal_hours"] > 0:
        return _rule_result(
            ctx["score_at"], threshold, "pool", "pooled monitor plus val at 0.15"
        )
    return rule_val_015(ctx, probe_cache)


def rule_quantile(ctx, probe_cache=None):
    # threshold = a percentile of selection interictal scores, the percentile set by the budget (see budget_exceedance_rate)
    # (real scores bunch up over time, so in practice this lands low)
    lookback = lookback_seconds(ctx["sop_seconds"])
    need = vote_need(lookback)
    rate = budget_exceedance_rate(lookback, need, FP_MAX_PER_HOUR)
    values = selection_interictal_values(ctx, ctx["t1"], ctx["t2"])
    select_set = "val"
    if len(values) == 0 and ctx["monitor_start"] is not None:
        values = selection_interictal_values(ctx, ctx["monitor_start"], ctx["monitor_end"])
        select_set = "monitor"
    if len(values) == 0:
        return _rule_result(
            ctx["score_at"], math.inf, "none", "no selection interictal scores"
        )
    threshold = empirical_quantile(values, 1.0 - rate)
    note = f"quantile at exceedance rate {rate:.4f} (level {1.0 - rate:.4f})"
    return _rule_result(ctx["score_at"], threshold, select_set, note)


def rule_smooth(ctx, probe_cache=None):
    window = lookback_seconds(ctx["sop_seconds"])
    smoothed = smooth_scores(ctx["score_at"], window)
    threshold, select_set, _info = pick_with_fallback(
        ctx, smoothed, FP_MAX_PER_HOUR, probe_cache=probe_cache
    )
    return _rule_result(
        smoothed,
        threshold,
        select_set,
        f"trailing mean over {window}s, then val budget 0.15",
    )


def rule_hysteresis(ctx, probe_cache=None):
    high, select_set, _info = pick_with_fallback(
        ctx, ctx["score_at"], FP_MAX_PER_HOUR, probe_cache=probe_cache
    )
    low, _low_set, _low_info = pick_with_fallback(
        ctx, ctx["score_at"], HYSTERESIS_LOW_FP_MAX_PER_HOUR, probe_cache=probe_cache
    )
    if math.isinf(low):
        low = high
    latched = latch_scores(ctx["score_at"], high, low)
    note = "hysteresis: enter above high (0.15 budget), exit below low (1.5 budget)"
    return _rule_result(latched, 0.5, select_set, note)


RULES = (
    ("val_0_15", rule_val_015),
    ("val_0_05", rule_val_005),
    ("vote_supremum", rule_vote_supremum),
    ("pool_0_15", rule_pool_015),
    ("quantile", rule_quantile),
    ("smooth", rule_smooth),
    ("hysteresis", rule_hysteresis),
)
