import torch
from torch.nn import BCEWithLogitsLoss

from core.config import LOCKED

# adam recipe and gradient clip follow the EvoBrain clone, MIT licence, https://github.com/Kotoge/EvoBrain
MAX_GRAD_NORM = 5.0


def make_optimizer(model):
    return torch.optim.Adam(
        model.parameters(),
        lr=LOCKED["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=LOCKED["weight_decay"],
    )


def one_train_step(
    model, optimizer, input_seq, seq_lengths, adj, labels, pos_weight, loss_fn=None
):
    if pos_weight <= 0:
        raise ValueError(f"pos_weight has to be > 0, we got {pos_weight}")
    model.train()
    optimizer.zero_grad()
    logits = model_logits(model, input_seq, seq_lengths, adj)
    labels = labels.float().reshape(-1)
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            f"logits and labels need the same batch, we got {logits.shape[0]} and {labels.shape[0]}"
        )
    if loss_fn is None:
        weight = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
        loss = BCEWithLogitsLoss(pos_weight=weight)(logits, labels)
    else:
        loss = loss_fn(logits, labels, pos_weight)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    optimizer.step()
    return float(loss.detach())


def model_logits(model, input_seq, seq_lengths, adj):
    out = model(input_seq, seq_lengths, adj)
    # evobrain returns (logits, hidden), a tiny test model can return logits only
    if type(out) is tuple:
        logits = out[0]
    else:
        logits = out
    return logits.reshape(-1)
