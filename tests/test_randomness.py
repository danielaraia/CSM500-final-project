import random

import numpy as np
import torch

from core.randomness import seed_everything


def test_seed_everything_repeats_model_initialization_and_rngs():
    seed_everything(321)
    first_model = torch.nn.Linear(4, 2)
    first_python = random.random()
    first_numpy = np.random.random()

    seed_everything(321)
    second_model = torch.nn.Linear(4, 2)
    second_python = random.random()
    second_numpy = np.random.random()

    torch.testing.assert_close(first_model.weight, second_model.weight)
    torch.testing.assert_close(first_model.bias, second_model.bias)
    assert first_python == second_python
    assert first_numpy == second_numpy
