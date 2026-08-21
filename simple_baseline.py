import numpy as np
from scipy.optimize import minimize

from alarms import choose_operating_threshold, score_map
from pos_weight import weight_from_class_numbers


def node_mean_energy(features):
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError(
            "features need 4 axes (windows, seconds, nodes, bins), "
            f"we got {values.ndim}"
        )
    return values.mean(axis=(1, 3))


def zscore_from_train(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"train features need 2 axes, we got {x.ndim}")
    if x.shape[0] < 2:
        raise ValueError(
            f"need at least 2 train windows to z score, we got {x.shape[0]}"
        )
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    index = 0
    while index < std.shape[0]:
        if std[index] <= 0.0:
            raise ValueError(
                f"train feature {index} has no spread, cannot z score"
            )
        index += 1
    return mean, std


def apply_zscore(x, mean, std):
    x = np.asarray(x, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    std = np.asarray(std, dtype=np.float64).reshape(-1)
    if x.ndim != 2:
        raise ValueError(f"features need 2 axes, we got {x.ndim}")
    if mean.shape[0] != x.shape[1] or std.shape[0] != x.shape[1]:
        raise ValueError(
            "z score mean and std must match the number of features, "
            f"we got {mean.shape[0]}, {std.shape[0]}, {x.shape[1]}"
        )
    return (x - mean) / std


def fit_logistic(x, y, pos_weight):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.ndim != 2:
        raise ValueError(f"logistic x needs 2 axes, we got {x.ndim}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            "x and y need the same number of rows, "
            f"we got {x.shape[0]} and {y.shape[0]}"
        )
    if pos_weight <= 0:
        raise ValueError(f"pos_weight has to be > 0, we got {pos_weight}")
    n_pos = 0
    n_neg = 0
    index = 0
    while index < y.shape[0]:
        lab = y[index]
        if lab == 1.0:
            n_pos += 1
        elif lab == 0.0:
            n_neg += 1
        else:
            raise ValueError(f"class numbers have to be 0 or 1, we got {lab}")
        index += 1
    if n_pos < 1 or n_neg < 1:
        raise ValueError(
            "logistic needs both classes on train, "
            f"we got {n_pos} sop and {n_neg} interictal"
        )
    xb = _add_bias(x)
    row_weight = np.ones(y.shape[0], dtype=np.float64)
    index = 0
    while index < y.shape[0]:
        if y[index] == 1.0:
            row_weight[index] = pos_weight
        index += 1

    def loss_and_grad(weights):
        z = xb @ weights
        per = y * _softplus(-z) + (1.0 - y) * _softplus(z)
        loss = float(np.mean(row_weight * per))
        resid = row_weight * (_sigmoid(z) - y)
        grad = (xb.T @ resid) / float(y.shape[0])
        return loss, grad

    start = np.zeros(xb.shape[1], dtype=np.float64)
    result = minimize(loss_and_grad, start, jac=True, method="L-BFGS-B")
    if not result.success:
        raise ValueError(f"logistic fit failed: {result.message}")
    return result.x


def predict_logits(x, weights):
    x = np.asarray(x, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    xb = _add_bias(x)
    if xb.shape[1] != weights.shape[0]:
        raise ValueError(
            "weights length does not match features plus bias, "
            f"we got {weights.shape[0]} and {xb.shape[1]}"
        )
    return xb @ weights


def fit_on_train_windows(features, labels):
    strength = node_mean_energy(features)
    labels = np.asarray(labels).reshape(-1)
    pos_weight, _n_neg, _n_pos = weight_from_class_numbers(labels)
    mean, std = zscore_from_train(strength)
    x = apply_zscore(strength, mean, std)
    weights = fit_logistic(x, labels, pos_weight)
    return {
        "mean": mean,
        "std": std,
        "weights": weights,
        "pos_weight": float(pos_weight),
    }


def logits_for_windows(features, model):
    strength = node_mean_energy(features)
    x = apply_zscore(strength, model["mean"], model["std"])
    return predict_logits(x, model["weights"])


def threshold_score_map(val_ends, val_logits, monitor_ends, monitor_logits):
    score_at = score_map(val_ends, val_logits)
    monitor_map = score_map(monitor_ends, monitor_logits)
    for second in monitor_map:
        if second in score_at:
            raise ValueError(f"val and monitor both scored second {second}")
        score_at[second] = monitor_map[second]
    return score_at


def pick_threshold_from_val_or_monitor(
    val_ends,
    val_logits,
    monitor_ends,
    monitor_logits,
    seizures,
    duration_seconds,
    val_start,
    val_end,
):
    score_at = threshold_score_map(
        val_ends, val_logits, monitor_ends, monitor_logits
    )
    monitor_ends = np.asarray(monitor_ends).reshape(-1)
    fallback_start = None
    fallback_end = None
    if monitor_ends.shape[0] > 0:
        fallback_start = int(np.min(monitor_ends))
        fallback_end = int(np.max(monitor_ends)) + 1
    return choose_operating_threshold(
        score_at,
        seizures,
        duration_seconds,
        val_start,
        val_end,
        fallback_start=fallback_start,
        fallback_end=fallback_end,
    )


def _add_bias(x):
    ones = np.ones((x.shape[0], 1), dtype=np.float64)
    return np.concatenate([x, ones], axis=1)


def _softplus(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = z[pos] + np.log1p(np.exp(-z[pos]))
    out[~pos] = np.log1p(np.exp(z[~pos]))
    return out


def _sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out
