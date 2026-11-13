"""Load verified source-only manifests and prepare images without cropping.

Training reads only train.csv and validation.csv. Raw images stay unchanged;
each image's bytes are checked against its manifest before preprocessing.
"""

import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path, PurePosixPath

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
BACKGROUND = tuple(round(x * 255) for x in MEAN)


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resize_with_padding(image):
    """Honor orientation, flatten transparency, and keep the entire image."""
    image = ImageOps.exif_transpose(image)
    if "A" in image.getbands() or "transparency" in image.info:
        rgba = image.convert("RGBA")
        rgb = Image.new("RGB", image.size, BACKGROUND)
        rgb.paste(rgba, mask=rgba.getchannel("A"))
    else:
        rgb = image.convert("RGB")
    resized = ImageOps.contain(rgb, (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), BACKGROUND)
    canvas.paste(resized, ((IMAGE_SIZE - resized.width) // 2,
                          (IMAGE_SIZE - resized.height) // 2))
    return canvas


def prepare_image(image):
    """Return the 3 x 224 x 224 normalized tensor expected by our model."""
    return TF.normalize(TF.to_tensor(resize_with_padding(image)), MEAN, STD)


def load_source_manifests(directory, categories_path):
    """Validate file fingerprints and reject any target/test training input."""
    directory = Path(directory)
    categories = json.loads(Path(categories_path).read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if sha256_file(categories_path) != summary["categories_sha256"]:
        raise ValueError("Category configuration changed; rebuild the split manifests")
    classes = categories["classes"]
    if [item["id"] for item in classes] != list(range(len(classes))) or not classes:
        raise ValueError("Category IDs must be consecutive from zero")
    datasets = {}
    seen = {"relative_path": set(), "sha256": set(), "group_id": set()}
    for split in ("train", "validation"):
        path = directory / f"{split}.csv"
        if sha256_file(path) != summary["manifest_sha256"][path.name]:
            raise ValueError(f"Manifest fingerprint mismatch: {path.name}")
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != summary["counts"][split]:
            raise ValueError(f"Manifest count mismatch: {split}")
        for row in rows:
            relative = PurePosixPath(row["relative_path"])
            if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
                raise ValueError("Unsafe image path")
            class_id = int(row["class_id"])
            if not 0 <= class_id < len(classes):
                raise ValueError("Unknown category ID")
            category = classes[class_id]["plantvillage"]
            if (row["dataset"] != "plantvillage" or row["source_split"] != "train"
                    or row["split"] != split or row["source_category"] != category
                    or row["class_name"] != category
                    or relative.parts[:3] != ("plantvillage", "raw", "color")):
                raise ValueError("Training/validation must use the mapped PlantVillage training pool")
        if {int(r["class_id"]) for r in rows} != set(range(len(classes))):
            raise ValueError(f"Missing category coverage in {split}")
        for key in seen:
            values = {r[key] for r in rows}
            if values & seen[key]:
                raise ValueError(f"Training/validation overlap: {key}")
            seen[key].update(values)
        datasets[split] = rows
    return datasets, categories, summary


def trial_subset(rows, per_class, seed):
    """Choose a repeatable per-category sample for debugging, not final scores."""
    if per_class < 1:
        raise ValueError("Trial sample size must be positive")
    ordered = sorted(rows, key=lambda r: hashlib.sha256(
        f"{seed}:{r['relative_path']}".encode()).hexdigest())
    counts = Counter()
    selected = []
    for row in ordered:
        label = int(row["class_id"])
        if counts[label] < per_class:
            selected.append(row)
            counts[label] += 1
    return sorted(selected, key=lambda r: r["relative_path"])


class PlantImages(Dataset):
    """Read the selected images without changing or silently skipping files."""

    def __init__(self, rows, data_root):
        self.rows = rows
        self.root = Path(data_root).expanduser().resolve()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = (self.root / row["relative_path"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Image path escapes the data directory")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != row["sha256"]:
            raise ValueError(f"Image changed since the audit: {row['relative_path']}")
        with Image.open(io.BytesIO(content)) as image:
            tensor = prepare_image(image)
        return tensor, int(row["class_id"])
