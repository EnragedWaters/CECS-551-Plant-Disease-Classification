"""Build the matched frozen-feature and fine-tuned MobileNetV3-Small models.

Load only the locally downloaded, fingerprinted ImageNet weights. Replace the
whole ImageNet classifier with a single linear layer over 576 pooled features.
"""

import hashlib

import torch
from torch import nn
from torchvision.models import mobilenet_v3_small

from .data import sha256_file


WEIGHTS_SHA256 = "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f"


def build_model(class_count, weights_path, seed):
    if class_count < 2:
        raise ValueError("At least two categories are required")
    if sha256_file(weights_path) != WEIGHTS_SHA256:
        raise ValueError("Pretrained weights do not match the pinned IMAGENET1K_V1 file")
    # The isolated CPU RNG gives both modes identical new classifier values
    # without disturbing the caller's random-number stream.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = mobilenet_v3_small(weights=None)
        model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        model.classifier = nn.Linear(model.classifier[0].in_features, class_count)
    set_training_mode(model, fine_tune=False)
    return model


def set_training_mode(model, fine_tune):
    """Choose trainable weights and prevent batch-norm statistics from drifting."""
    model.train()
    model.features.requires_grad_(fine_tune)
    model.classifier.requires_grad_(True)
    if not fine_tune:
        model.features.eval()
    # BatchNorm's affine weights can learn during fine-tuning, but its running
    # means/variances stay fixed in both modes, as specified in the proposal.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def state_fingerprint(state):
    """Fingerprint weights AND buffers to verify the frozen extractor stays fixed."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def batch_norm_state(model):
    return {name: value for name, value in model.state_dict().items()
            if name.endswith(("running_mean", "running_var", "num_batches_tracked"))}
