import numpy as np
import pytest
import torch


def test_evobrain_batch_does_not_change_logits():
    pytest.importorskip("torch_geometric")
    pytest.importorskip("mamba_ssm")
    if not torch.cuda.is_available():
        pytest.skip("needs cuda")

    from core.randomness import seed_everything
    from model.evobrain_load import load_evobrain
    from model.train_loop import score_windows

    rng = np.random.default_rng(123)
    features = rng.normal(size=(2, 60, 8, 128)).astype(np.float32)
    graphs = rng.normal(size=(2, 60, 8, 8)).astype(np.float32)
    seed_everything(123)
    model = load_evobrain(device="cuda").to("cuda")
    together = score_windows(
        model, features, graphs, batch_size=2, device="cuda"
    )
    separate = score_windows(
        model, features, graphs, batch_size=1, device="cuda"
    )
    torch.testing.assert_close(together, separate, rtol=1e-4, atol=2e-5)
