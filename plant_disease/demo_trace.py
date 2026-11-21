"""Observe real model activations without changing the forward computation.

Hooks copy outputs from three fixed feature blocks and average pooling. They
return nothing, are removed even on failure, and are checked against a normal
forward pass. Display colors share one symmetric scale across both models for
each stage; they are feature responses, not attention or disease explanations.
"""
import base64
import io

import numpy as np
import torch
from PIL import Image

# Fixed blocks/channels, selected by architecture position rather than appearance.
BLOCKS = ((0, 'early', 'Early features'), (4, 'middle', 'Middle features'),
          (12, 'late', 'Late features'))


def observe(model, tensor):
    """Return genuine outputs and prove hooks left the ordinary logits unchanged."""
    captures, handles = {}, []

    def recorder(key):
        def hook(module, arguments, output):
            captures[key] = output.detach().cpu().clone()
            # Deliberately return None: returning a tensor would replace output.
        return hook

    with torch.inference_mode():
        baseline = model(tensor).detach().clone()
        try:
            for index, key, _ in BLOCKS:
                handles.append(model.features[index].register_forward_hook(recorder(key)))
            handles.append(model.avgpool.register_forward_hook(recorder('pooled')))
            logits = model(tensor)
        finally:
            for handle in handles:
                handle.remove()
        if not torch.equal(baseline, logits):
            raise RuntimeError('Observation changed the model output')
    if not torch.isfinite(logits).all() or any(not torch.isfinite(v).all() for v in captures.values()):
        raise ValueError('Non-finite model output or feature map')
    return logits, captures


def color_image(values, limit, size=None):
    """Blue = negative, pale = zero, orange = positive; no per-image rescaling."""
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or not np.isfinite(limit) or limit <= 0:
        raise ValueError('Feature display requires finite values and a positive scale')
    signed = np.clip(values / limit, -1, 1)
    neutral = np.array([244, 245, 240])
    negative, positive = np.array([43, 98, 177]), np.array([207, 102, 33])
    target = np.where(signed[..., None] < 0, negative, positive)
    rgb = neutral + np.abs(signed[..., None]) * (target - neutral)
    image = Image.fromarray(np.rint(rgb).astype(np.uint8))
    if size:
        image = image.resize(size, Image.Resampling.NEAREST)
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def describe_pair(captures):
    """Make comparable visuals using the same channels and scale for both models."""
    result = {mode: [] for mode in captures}
    for _, key, title in BLOCKS:
        tensors = {mode: stages[key][0] for mode, stages in captures.items()}
        if len({tuple(t.shape) for t in tensors.values()}) != 1:
            raise ValueError('Feature shapes differ between models')
        channels = list(range(min(4, next(iter(tensors.values())).shape[0])))
        limit = max(float(t[channels].abs().max()) for t in tensors.values()) or 1.0
        for mode, tensor in tensors.items():
            result[mode].append({'key': key, 'title': title, 'shape': list(tensor.shape),
                'scale_limit': limit, 'channels': channels,
                'maps': [color_image(tensor[c].numpy(), limit, (128,128)) for c in channels]})
    pooled = {mode: stages['pooled'][0].flatten() for mode, stages in captures.items()}
    if any(v.numel() != 576 for v in pooled.values()):
        raise ValueError('Expected 576 pooled features from the original model')
    limit = max(float(v.abs().max()) for v in pooled.values()) or 1.0
    for mode, vector in pooled.items():
        result[mode].append({'key': 'pooled', 'title': 'Summarize the feature maps',
            'shape': [576], 'scale_limit': limit, 'values': vector.tolist(),
            'maps': [color_image(vector.numpy().reshape(24,24), limit, (240,240))],
            'layout_note': '576 real averages arranged in a 24 × 24 grid for display; these are not image locations.'})
    return result


def learning_values(model, captures, logits, true_class):
    """Reconstruct the original pooling and linear head for a numerical lesson.

    Double precision makes the displayed sums readable and stable. The actual
    prediction still comes from the untouched float32 forward pass. We verify
    all channels/classes, not just the few expanded in the interface.
    """
    late = captures['late'][0].double()
    pooled = captures['pooled'][0].flatten().double()
    if tuple(late.shape) != (576, 7, 7) or pooled.numel() != 576:
        raise ValueError('Teaching lesson requires the original 576 by 7 by 7 features')
    means = late.mean(dim=(1, 2))
    if not torch.allclose(means, pooled, atol=2e-6, rtol=1e-5):
        raise ValueError('Pooling reconstruction does not match the observed averages')
    if not isinstance(model.classifier, torch.nn.Linear):
        raise ValueError('Teaching lesson requires the original single linear classifier')
    weights = model.classifier.weight.detach().cpu().double()
    bias = model.classifier.bias.detach().cpu().double()
    products = weights * pooled.unsqueeze(0)
    reconstructed = products.sum(dim=1) + bias
    actual = logits[0].detach().cpu().double()
    if not torch.isfinite(reconstructed).all() or not torch.allclose(reconstructed, actual, atol=2e-5, rtol=1e-4):
        raise ValueError('Classifier reconstruction does not match the observed logits')
    ranked = torch.argsort(actual, descending=True).tolist()
    roles = [('Prediction', ranked[0]), ('Runner-up', ranked[1]), ('Dataset label', true_class)]
    classes = []
    for category in dict.fromkeys(category for _, category in roles):
        contribution = products[category]
        indexes = torch.argsort(contribution.abs(), descending=True)[:5].tolist()
        remaining = contribution.clone()
        remaining[indexes] = 0
        classes.append({'class_id': category,
            'roles': [name for name, i in roles if i == category],
            'terms': [{'channel': i, 'value': float(pooled[i]),
                       'weight': float(weights[category, i]), 'product': float(contribution[i])}
                      for i in indexes],
            'remaining_count': pooled.numel() - len(indexes),
            'remaining_sum': float(remaining.sum()), 'bias': float(bias[category]),
            'reconstructed_logit': float(reconstructed[category]), 'actual_logit': float(actual[category])})
    return {'pooling': [{'channel': i, 'grid': late[i].tolist(), 'sum': float(late[i].sum()),
                        'mean': float(means[i]), 'observed': float(pooled[i])} for i in range(4)],
            'classifier': classes,
            'pooling_max_error': float((means-pooled).abs().max()),
            'classifier_max_error': float((reconstructed-actual).abs().max()),
            'all_576_averages_verified': True, 'all_logits_verified': True}


def choose_batch(rows, class_count):
    """One example per class per dataset, by lowest byte hash then relative path.

No prediction, confidence or correctness is consulted. This balanced teaching
sample is not representative of either original test population.
"""
    chosen = []
    for label in range(class_count):
        pair = {}
        for split, dataset in [('source_test','plantvillage'), ('target_test','plantdoc')]:
            candidates = [r for r in rows[split].values() if int(r['class_id']) == label]
            if not candidates:
                raise ValueError(f'Missing category {label} in {split}')
            pair[dataset] = min(candidates, key=lambda r:(r['sha256'],r['relative_path']))
        chosen.append(pair)
    return chosen
