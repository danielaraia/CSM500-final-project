import torch
import torch.nn.functional as F

# focal loss from Lin et al 2017
# gamma 2; the positive class alpha is the existing pos_weight,
# same role it plays in the original weighted bce
FOCAL_GAMMA = 2.0
LOSS_NAMES = ("weighted_bce", "focal")


def focal_loss_with_logits(logits, labels, pos_weight, gamma=FOCAL_GAMMA):
    if pos_weight <= 0:
        raise ValueError(f"pos_weight has to be > 0, we got {pos_weight}")
    if gamma < 0:
        raise ValueError(f"gamma has to be >= 0, we got {gamma}")
    logits = logits.reshape(-1)
    labels = labels.float().reshape(-1)
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            "logits and labels need the same batch, "
            f"we got {logits.shape[0]} and {labels.shape[0]}"
        )
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * labels + (1.0 - p) * (1.0 - labels)
    alpha = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    weight = alpha * labels + (1.0 - labels)
    loss = weight * torch.pow(1.0 - p_t, gamma) * bce
    return loss.mean()


def make_loss_fn(name):
    # none means one_train_step keeps its own weighted bce
    if name == "weighted_bce":
        return None
    if name == "focal":
        def focal(logits, labels, pos_weight):
            return focal_loss_with_logits(logits, labels, pos_weight)

        return focal
    raise ValueError(f"loss has to be weighted_bce or focal, we got {name!r}")
