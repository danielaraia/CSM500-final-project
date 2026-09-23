import torch

from config import LOCKED

# train slice only: small gaussian jitter plus a per window amplitude scale.
# the main config has data_augment False and the evobrain augment flag only reaches their dataloader, which this pipeline never uses, so these live here
AUGMENT_NAMES = ("none", "jitter_scale")
JITTER_SIGMA = 0.01
SCALE_LOW = 0.9
SCALE_HIGH = 1.1


def jitter_scale(features, generator):
    scale = torch.empty(
        features.shape[0], 1, 1, 1, dtype=features.dtype, device=features.device
    )
    scale.uniform_(SCALE_LOW, SCALE_HIGH, generator=generator)
    noise = torch.randn(
        features.shape,
        generator=generator,
        dtype=features.dtype,
        device=features.device,
    )
    return features * scale + JITTER_SIGMA * noise


def make_augment_fn(name, device=None):
    # none means the default path: train batches go to the model untouched
    if name == "none":
        return None
    if name != "jitter_scale":
        raise ValueError(f"augment has to be none or jitter_scale, we got {name!r}")
    # a dedicated generator so augmentation never eats the global rng stream
    generator = torch.Generator(device=device if device is not None else "cpu")
    generator.manual_seed(LOCKED["seed"])

    def augment(features):
        return jitter_scale(features, generator)

    return augment
