"""Train on PlantVillage and select checkpoints using source validation only.

Short GPU feasibility check from the repository root:
    python -m plant_disease.train --trial --output-dir outputs/trials/first

Without --trial, the same implementation uses all source train/validation
images (ten epochs by default). Test manifests are never opened by this runner.
Outputs include run settings, selected filenames, epoch metrics and checkpoints.
"""

import argparse
import csv
import json
import os
import platform
import random
import time
from pathlib import Path

# Required for deterministic CUDA matrix multiplication; set before CUDA starts.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torchvision
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from .data import (BACKGROUND, MEAN, STD, PlantImages, load_source_manifests,
                   sha256_file, trial_subset)
from .model import (WEIGHTS_SHA256, batch_norm_state, build_model,
                    set_training_mode, state_fingerprint)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_epoch(model, loader, device, class_count, optimizer=None):
    """Return image-weighted loss and metrics; validation never computes gradients."""
    if optimizer is None:
        model.eval()
    total_loss = 0.0
    actual, predicted = [], []
    synchronize(device)
    start = time.perf_counter()
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; stop rather than save invalid results")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            actual.extend(labels.detach().cpu().tolist())
            predicted.extend(logits.detach().argmax(1).cpu().tolist())
    synchronize(device)
    if not actual:
        raise ValueError("Empty data loader")
    return {"images": len(actual), "loss": total_loss / len(actual),
            "accuracy": accuracy_score(actual, predicted),
            "macro_f1": f1_score(actual, predicted, labels=list(range(class_count)),
                                 average="macro", zero_division=0),
            "seconds": time.perf_counter() - start}


def make_optimizer(model, head_lr, backbone_lr, weight_decay, fine_tune):
    groups = [{"params": model.classifier.parameters(), "lr": head_lr}]
    if fine_tune:
        groups.append({"params": model.features.parameters(), "lr": backbone_lr})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def train_mode(mode, args, rows, class_count, device, run_metadata):
    """Use matched initialization, image order, budgets and checkpoint rules."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = build_model(class_count, args.weights, args.seed).to(device)
    feature_start = state_fingerprint(model.features.state_dict())
    classifier_start = state_fingerprint(model.classifier.state_dict())
    bn_start = state_fingerprint(batch_norm_state(model))
    run_dir = args.output_dir / mode
    run_dir.mkdir()
    train_generator = torch.Generator()
    train_loader = DataLoader(PlantImages(rows["train"], args.data_root), batch_size=args.batch_size,
                              shuffle=True, generator=train_generator, num_workers=args.workers,
                              pin_memory=device.type == "cuda")
    val_loader = DataLoader(PlantImages(rows["validation"], args.data_root), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    best_loss, best_epoch, bad_epochs = float("inf"), None, 0
    history = []
    optimizer = None
    for epoch in range(1, args.epochs + 1):
        fine_tune = mode == "finetune" and epoch > args.warmup_epochs
        set_training_mode(model, fine_tune)
        if epoch in (1, args.warmup_epochs + 1):
            # Reset head optimizer moments at the same boundary for BOTH modes.
            optimizer = make_optimizer(model, args.head_lr, args.backbone_lr,
                                       args.weight_decay, fine_tune)
        train_generator.manual_seed(args.seed + epoch)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_metrics = run_epoch(model, train_loader, device, class_count, optimizer)
        validation_metrics = run_epoch(model, val_loader, device, class_count)
        features_now = state_fingerprint(model.features.state_dict())
        if state_fingerprint(batch_norm_state(model)) != bn_start:
            raise RuntimeError("Batch-normalization statistics changed unexpectedly")
        if not fine_tune and features_now != feature_start:
            raise RuntimeError("Frozen feature weights or buffers changed")
        if fine_tune and features_now == feature_start:
            raise RuntimeError("Fine-tuning did not change any feature weights")
        if state_fingerprint(model.classifier.state_dict()) == classifier_start:
            raise RuntimeError("Classifier weights did not learn")
        record = {"epoch": epoch, "stage": "warmup" if epoch <= args.warmup_epochs else mode,
                  "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                  "train": train_metrics, "validation": validation_metrics,
                  "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None,
                  "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2 if device.type == "cuda" else None}
        history.append(record)
        save_json(run_dir / "history.json", history)
        # Select among post-warmup epochs in BOTH modes. Validation loss, never
        # external test accuracy, decides the saved model and early stopping.
        eligible = epoch > args.warmup_epochs
        improved = eligible and validation_metrics["loss"] < best_loss
        if improved:
            best_loss, best_epoch, bad_epochs = validation_metrics["loss"], epoch, 0
        elif eligible:
            bad_epochs += 1
        checkpoint = {"model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                      "optimizer_state": optimizer.state_dict(), "epoch": epoch, "mode": mode,
                      "trial": args.trial, "class_count": class_count,
                      "metadata": run_metadata, "validation_loss": validation_metrics["loss"]}
        torch.save(checkpoint, run_dir / "last.pt")
        if improved:
            torch.save(checkpoint, run_dir / "best.pt")
        print(f"{mode} epoch {epoch}/{args.epochs} ({record['stage']}): "
              f"train loss={train_metrics['loss']:.4f}; validation loss={validation_metrics['loss']:.4f}; "
              f"train={train_metrics['seconds']:.1f}s, validation={validation_metrics['seconds']:.1f}s", flush=True)
        if eligible and args.patience and bad_epochs >= args.patience:
            break
    result = {"mode": mode, "trial": args.trial, "epochs_completed": len(history),
              "best_epoch": best_epoch, "best_validation_loss": best_loss,
              "initial_feature_sha256": feature_start, "initial_classifier_sha256": classifier_start,
              "final_feature_sha256": features_now, "batch_norm_statistics_unchanged": True,
              "total_parameters": sum(p.numel() for p in model.parameters()), "history": history}
    save_json(run_dir / "result.json", result)
    del checkpoint, optimizer, model, train_loader, val_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path.home() / "datasets/cecs551-plant-disease")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--splits-dir", type=Path, default=Path("outputs/splits"))
    parser.add_argument("--categories", type=Path, default=Path("configs/categories.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("both", "frozen", "finetune"), default="both")
    parser.add_argument("--trial", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=551)
    parser.add_argument("--head-lr", type=float, default=0.001)
    parser.add_argument("--backbone-lr", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    args.epochs = args.epochs if args.epochs is not None else (2 if args.trial else 10)
    args.data_root = args.data_root.expanduser()
    args.weights = (args.weights or args.data_root / "weights/mobilenet_v3_small-047dcff4.pth").expanduser()
    if args.warmup_epochs < 1 or args.epochs <= args.warmup_epochs:
        parser.error("Need at least one warmup epoch and one post-warmup epoch")
    if args.batch_size < 1 or args.workers < 0 or args.patience < 0:
        parser.error("Invalid batch size, worker count or patience")
    if args.head_lr <= 0 or args.backbone_lr <= 0 or args.weight_decay < 0:
        parser.error("Learning rates must be positive and weight decay nonnegative")
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new run name")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; use --device cpu explicitly if needed")
    if sha256_file(args.weights) != WEIGHTS_SHA256:
        raise ValueError("Pretrained weight fingerprint mismatch")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    rows, categories, split_summary = load_source_manifests(args.splits_dir, args.categories)
    if args.trial:
        rows = {split: trial_subset(records, 32 if split == "train" else 8, args.seed)
                for split, records in rows.items()}
    args.output_dir.mkdir(parents=True)
    for split, records in rows.items():
        with (args.output_dir / f"selected_{split}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    metadata = {"scope": "feasibility_trial_not_final_results" if args.trial else "source_training_and_validation",
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "counts": {k: len(v) for k, v in rows.items()}, "categories": categories["classes"],
                "software": {"python": platform.python_version(), "torch": str(torch.__version__),
                             "torchvision": str(torchvision.__version__)},
                "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                "weights_sha256": WEIGHTS_SHA256, "split_summary": split_summary,
                "code_sha256": {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                "selected_manifest_sha256": {split: sha256_file(args.output_dir / f"selected_{split}.csv") for split in rows},
                "preprocessing": {"size": 224, "resize": "bilinear, preserve aspect ratio, pad",
                                  "background_rgb": BACKGROUND, "mean": MEAN, "std": STD,
                                  "exif_orientation": True, "augmentation": "none"},
                "precision": "float32", "deterministic_algorithms": True,
                "checkpoint_rule": "minimum source validation cross-entropy after warmup",
                "test_manifests_opened": False}
    save_json(args.output_dir / "run.json", metadata)
    modes = ("frozen", "finetune") if args.mode == "both" else (args.mode,)
    results = {mode: train_mode(mode, args, rows, len(categories["classes"]), device, metadata)
               for mode in modes}
    if len(results) == 2:
        for key in ("initial_feature_sha256", "initial_classifier_sha256"):
            if results["frozen"][key] != results["finetune"][key]:
                raise RuntimeError("Model initializations did not match")
    save_json(args.output_dir / "comparison.json", results)
    print(f"Completed {'feasibility trial' if args.trial else 'source training'}: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
