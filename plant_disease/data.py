"""Load verified training manifests and prepare images with optional augmentation.

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
from torchvision import transforms
from torchvision.transforms import functional as TF


IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
BACKGROUND = tuple(round(x * 255) for x in MEAN)
def augmentation_recipe(name):
    """Describe exactly the random changes applied to TRAINING images only."""
    if name == "none":
        return "none"
    if name != "basic-v1":
        raise ValueError(f"Unknown augmentation: {name}")
    return {
        "name": name, "applies_to": "training only",
        "order": ["random_resized_crop", "horizontal_flip", "rotation", "color_jitter"],
        "random_resized_crop": {"size": IMAGE_SIZE, "scale": [0.7, 1.0],
                                "ratio": [0.75, 4 / 3], "interpolation": "bilinear", "antialias": True},
        "horizontal_flip": {"p": 0.5},
        "rotation": {"degrees": [-20, 20], "interpolation": "nearest",
                     "expand": False, "fill": list(BACKGROUND)},
        "color_jitter": {"brightness": 0.25, "contrast": 0.25, "saturation": 0.25, "hue": 0.05},
    }


def make_augmentation(name="none"):
    """Build the transform from the same recipe that is saved in run.json."""
    recipe = augmentation_recipe(name)
    if recipe == "none":
        return None
    crop = recipe["random_resized_crop"]
    rotation = recipe["rotation"]
    return transforms.Compose([
        transforms.RandomResizedCrop(crop["size"], scale=tuple(crop["scale"]),
            ratio=tuple(crop["ratio"]), interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=crop["antialias"]),
        transforms.RandomHorizontalFlip(**recipe["horizontal_flip"]),
        transforms.RandomRotation(rotation["degrees"],
            interpolation=transforms.InterpolationMode.NEAREST,
            expand=rotation["expand"], fill=tuple(rotation["fill"])),
        transforms.ColorJitter(**recipe["color_jitter"]),
    ])


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def oriented_rgb(image):
    """Honor orientation and flatten transparency before any color changes."""
    image = ImageOps.exif_transpose(image)
    if "A" in image.getbands() or "transparency" in image.info:
        rgba = image.convert("RGBA")
        rgb = Image.new("RGB", image.size, BACKGROUND)
        rgb.paste(rgba, mask=rgba.getchannel("A"))
    else:
        rgb = image.convert("RGB")
    return rgb


def resize_with_padding(image):
    """Keep the entire image and pad to the model input size."""
    rgb = oriented_rgb(image)
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
            '''
            category = classes[class_id]["plantvillage"]
            if (row["dataset"] != "plantvillage" or row["source_split"] != "train"
                    or row["split"] != split or row["source_category"] != category
                    or row["class_name"] != category
                    or relative.parts[:3] != ("plantvillage", "raw", "color")):
                raise ValueError("Training/validation must use the mapped PlantVillage training pool")
            '''
            # Validate the dataset before looking up its category mapping.
            # Otherwise malformed input crashes with an unexplained KeyError.
            dataset = row["dataset"]
            if dataset not in ("plantvillage", "plantdoc") or dataset not in classes[class_id]:
                raise ValueError("Training dataset is not allowed or has no category mapping")
            category = classes[class_id][dataset]
            parent = (("plantvillage", "raw", "color", category) if dataset == "plantvillage"
                      else ("plantdoc", "train", category))
            if (row["source_split"] != "train" or row["split"] != split
                    or row["source_category"] != category
                    or row["class_name"] != classes[class_id]["plantvillage"]
                    or relative.parts[:-1] != parent):
                raise ValueError("Training/validation must use the mapped training images")

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

    def __init__(self, rows, data_root, transform=None):
        self.rows = rows
        self.root = Path(data_root).expanduser().resolve()
        self.transform = transform

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
            image = oriented_rgb(image)
            if self.transform is not None:
                image = self.transform(image)
            tensor = prepare_image(image)

        return tensor, int(row["class_id"])
