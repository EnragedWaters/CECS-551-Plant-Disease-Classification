"""Reload saved models and reproduce their recorded validation scores.

Run: python -m plant_disease.verify_training --run-dir outputs/runs/source-seed-552
Reads only the recorded train/validation manifests and validation images. Does not
train, change checkpoints, or open final test lists. Writes a small verification
record inside the ignored run folder. Uses the recorded device and batch size.
"""

import argparse
import json
from pathlib import Path

# Import first: train configures deterministic CUDA before the GPU is used.
from .train import run_epoch
import torch
from torch.utils.data import DataLoader

from .data import PlantImages, load_source_manifests, sha256_file
from .model import batch_norm_state, build_model, state_fingerprint
from .summarize_training import collect_summary


def verify(run_dir, data_root, splits_dir, categories_path):
    summary = collect_summary(run_dir)
    metadata = json.loads((run_dir / "run.json").read_text())
    args = metadata["arguments"]
    for name in ("__init__.py", "data.py", "model.py", "train.py"):
        if sha256_file(Path(__file__).parent / name) != metadata["code_sha256"][name]:
            raise ValueError(f"Training implementation changed: {name}")
    rows, categories, split_summary = load_source_manifests(
        splits_dir, categories_path, args.get("include_plantdoc_train", False))
    if split_summary != metadata["split_summary"]:
        raise ValueError("Split configuration changed after training")
    if {key: len(value) for key, value in rows.items()} != metadata["counts"]:
        raise ValueError("This is not the recorded full validation pool")
    for split in rows:
        if sha256_file(splits_dir / f"{split}.csv") != sha256_file(run_dir / f"selected_{split}.csv"):
            raise ValueError("Selected images do not match the current full training/validation lists")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args["device"])
    loader = DataLoader(PlantImages(rows["validation"], data_root), batch_size=args["batch_size"],
                        shuffle=False, num_workers=args["workers"], pin_memory=device.type == "cuda")
    checks = {}
    for mode, result in summary["models"].items():
        path = run_dir / mode / "best.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (checkpoint["trial"] or checkpoint["epoch"] != result["best_epoch"]
                or checkpoint["mode"] != mode
                or json.loads(json.dumps(checkpoint["metadata"])) != metadata):
            raise ValueError("Checkpoint does not match the recorded run")
        model = build_model(len(categories["classes"]),
                            data_root / "weights/mobilenet_v3_small-047dcff4.pth", args["seed"])
        original_bn = state_fingerprint(batch_norm_state(model))
        original_features = state_fingerprint(model.features.state_dict())
        model.load_state_dict(checkpoint["model_state"])
        if state_fingerprint(batch_norm_state(model)) != original_bn:
            raise ValueError("Batch-normalization statistics changed")
        frozen = state_fingerprint(model.features.state_dict()) == original_features
        if frozen != (mode == "frozen"):
            raise ValueError("Feature weights do not match the intended learning mode")
        measured = run_epoch(model.to(device), loader, device, len(categories["classes"]))
        for key in ("images", "loss", "accuracy", "macro_f1"):
            if abs(measured[key] - result["validation"][key]) > 1e-7:
                raise ValueError(f"Reloaded score differs: {mode}, {key}")
        checks[mode] = {"epoch": checkpoint["epoch"], "reloaded_validation_matches_record": True,
                        "measured": measured, "best_checkpoint_sha256": sha256_file(path)}
        print(f"Verified {mode}: epoch {checkpoint['epoch']}, all {len(rows['validation'])} validation images", flush=True)
        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()
    (run_dir / "checkpoint-verification.json").write_text(json.dumps(checks, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path.home() / "datasets/cecs551-plant-disease")
    parser.add_argument("--splits-dir", type=Path, default=Path("outputs/splits"))
    parser.add_argument("--categories", type=Path, default=Path("configs/categories.json"))
    args = parser.parse_args()
    verify(args.run_dir, args.data_root.expanduser(), args.splits_dir, args.categories)


if __name__ == "__main__":
    main()
