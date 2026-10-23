"""Check that our Python packages and basic model execution work.

This script loads the required packages, checks GPU availability, and passes
one artificial all-black image through MobileNetV3-Small. It checks that the
model returns valid numbers, then prints the software versions and device used.

Nothing is downloaded, and no model training or accuracy testing takes place.
The model starts with random weights, so its category scores are meaningless.
A passing check confirms basic execution; real training speed and memory needs
still require a separate trial.

Run from the project folder with the project Python environment active:
    python scripts/check_environment.py --require-cuda

CUDA is the software support PyTorch uses to work with an NVIDIA GPU. The flag
requires a working GPU; without it, the script can fall back to the CPU.
"""

import argparse
import importlib
import importlib.metadata
import json
import platform
import sys

import torch
import torchvision


def main():
    # Read the optional command-line flag so a GPU setup problem cannot quietly
    # pass by running on the CPU when --require-cuda was requested.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-cuda", action="store_true", help="Fail if the NVIDIA GPU is unavailable."
    )
    args = parser.parse_args()

    # Check that the analysis and notebook tools load too. Pillow is imported as
    # PIL, and scikit-learn as sklearn. A missing/broken import stops the check.
    for module in ("numpy", "PIL", "pandas", "matplotlib", "sklearn", "ipykernel"):
        importlib.import_module(module)

    # Ask PyTorch whether it can use the GPU, then choose where to run the model.
    has_cuda = torch.cuda.is_available()
    if args.require_cuda and not has_cuda:
        raise SystemExit("CUDA is unavailable in this Python environment.")
    device = torch.device("cuda" if has_cuda else "cpu")

    # weights=None avoids downloading pretrained weights. This is a setup check,
    # not the model configuration or evidence for the proposed experiment.
    model = torchvision.models.mobilenet_v3_small(weights=None).to(device).eval()
    # eval() uses evaluation behavior; inference_mode() skips bookkeeping for
    # learning. The input dimensions are: 1 image, 3 color channels, 224 x 224
    # pixels. All zeros give us a black image without needing an image file.
    with torch.inference_mode():
        scores = model(torch.zeros(1, 3, 224, 224, device=device))
    # The unchanged model has 1,000 output categories. Check that we get one set
    # of scores with no NaN (invalid) or infinite values, not a correct prediction.
    if tuple(scores.shape) != (1, 1000) or not torch.isfinite(scores).all().item():
        raise RuntimeError("MobileNet forward-pass check failed.")

    # Print a readable setup report so we can identify the environment used and
    # troubleshoot differences between teammates' machines.
    result = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pillow", "pandas", "matplotlib", "scikit-learn", "ipykernel")
        },
        "forward_pass": "passed",
        "output_shape": list(scores.shape),
        "input": "one synthetic zero image, not dataset data",
        "pretrained_weights_downloaded": False,
        "training_performed": False,
    }
    if has_cuda:
        gpu = torch.cuda.get_device_properties(device)
        result["gpu_name"] = gpu.name
        result["gpu_total_memory_mib"] = round(gpu.total_memory / 1024**2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    # Run the check when launched as a script, rather than when imported.
    main()
