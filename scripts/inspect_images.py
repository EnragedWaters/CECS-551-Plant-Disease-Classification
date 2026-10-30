"""Measure local image sizes and make reproducible picture sheets for Phase 3.

Run from the project with: python scripts/inspect_images.py
Use --data-root for another dataset location and --output-dir for the outputs.
The defaults match prepare_data.py and write to Git-ignored outputs/image-audit.
This reads image headers, not a full pixel/duplicate/disease-label audit. Only
the displayed samples are decoded here. Raw files are never modified; no
downloads, model predictions, or training take place. CSV paths are relative.
"""

import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# Chosen before viewing, to include healthy and disease-labeled examples.
# These correspond to six proposed pairs in docs/category-mapping.md.
SAMPLE_PAIRS = [
    ("Apple scab", "Apple___Apple_scab", "Apple Scab Leaf"),
    ("Apple healthy", "Apple___healthy", "Apple leaf"),
    ("Grape black rot", "Grape___Black_rot", "grape leaf black rot"),
    ("Potato early blight", "Potato___Early_blight", "Potato leaf early blight"),
    ("Tomato early blight", "Tomato___Early_blight", "Tomato Early blight leaf"),
    ("Tomato healthy", "Tomato___healthy", "Tomato leaf"),
]


def write_csv(path, rows):
    """Keep a header even when no image-header errors were found."""
    if not rows:
        path.write_text("relative_path,error\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def inspect(data_root, output_dir):
    """Inventory every supported file; keep each supplied split distinguishable."""
    folders = [
        ("PlantVillage", "color", data_root / "plantvillage/raw/color"),
        ("PlantDoc", "train", data_root / "plantdoc/train"),
        ("PlantDoc", "test", data_root / "plantdoc/test"),
    ]
    for _, _, folder in folders:
        if not folder.is_dir():
            raise FileNotFoundError(f"Dataset folder missing: {folder}")
    rows, errors, summaries = [], [], []
    for dataset, split, folder in folders:
        paths = sorted(p for p in folder.rglob("*")
                       if p.is_file() and p.suffix.lower() in EXTENSIONS)
        subset = []
        for path in paths:
            try:
                with Image.open(path) as image:
                    width, height = image.size
                    mode = image.mode
                row = dict(dataset=dataset, split=split, category=path.parent.name,
                           relative_path=path.relative_to(data_root).as_posix(),
                           width=width, height=height, mode=mode)
                subset.append(row)
            except (OSError, ValueError) as exc:
                errors.append(dict(relative_path=path.relative_to(data_root).as_posix(),
                                   error=str(exc)))
        if not subset:
            raise RuntimeError(f"No readable image headers in {folder}")
        widths = [r["width"] for r in subset]
        heights = [r["height"] for r in subset]
        counts = Counter(r["category"] for r in subset)
        summaries.append(dict(
            dataset=dataset, split=split, files_found=len(paths), headers_read=len(subset),
            categories=len(counts), width_min=min(widths), width_median=float(np.median(widths)),
            width_max=max(widths), height_min=min(heights),
            height_median=float(np.median(heights)), height_max=max(heights),
            nonsquare=sum(r["width"] != r["height"] for r in subset),
            modes=dict(Counter(r["mode"] for r in subset)),
            category_min=min(counts.values()), category_max=max(counts.values()),
        ))
        rows.extend(subset)
        print(f"{dataset} {split}: {len(subset):,} headers read", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "image-inventory.csv", rows)
    write_csv(output_dir / "header-errors.csv", errors)
    counts = Counter((r["dataset"], r["split"], r["category"]) for r in rows)
    write_csv(output_dir / "category-counts.csv", [
        dict(dataset=d, split=s, category=c, count=n)
        for (d, s, c), n in sorted(counts.items())
    ])
    (output_dir / "size-summary.json").write_text(
        json.dumps(dict(partitions=summaries, header_errors=len(errors)), indent=2) + "\n",
        encoding="utf-8")
    return rows


def picture_sheets(rows, data_root, output_dir, seed):
    """Two samples per dataset/category, without filtering for picture quality.

    PlantDoc samples come only from train: these are descriptive illustrations,
    not target data for fitting/selecting the model. Test sizes are still counted.
    Category selection is deliberate, so the sample is not population-random.
    """
    samples = []
    for sheet in range(2):
        fig, axes = plt.subplots(3, 4, figsize=(14, 11))
        for row_index, (title, pv_label, pd_label) in enumerate(SAMPLE_PAIRS[sheet * 3:sheet * 3 + 3]):
            for ds_index, (dataset, label) in enumerate((("PlantVillage", pv_label), ("PlantDoc", pd_label))):
                candidates = [r for r in rows if r["dataset"] == dataset and r["category"] == label
                              and (dataset == "PlantVillage" or r["split"] == "train")]
                rng = random.Random(f"{seed}:{dataset}:{label}")
                chosen = rng.sample(candidates, 2)
                for sample_index, record in enumerate(chosen):
                    ident = f"S{sheet+1}-{row_index+1}-{ds_index*2+sample_index+1}"
                    ax = axes[row_index, ds_index*2+sample_index]
                    with Image.open(data_root / record["relative_path"]) as opened:
                        # Honor camera orientation for display; do not crop the image.
                        image = ImageOps.exif_transpose(opened).convert("RGB")
                        ax.imshow(image)
                    ax.set_title(f"{ident}: {dataset}\n{title} | {record['width']} × {record['height']}", fontsize=10)
                    ax.axis("off")
                    samples.append(dict(sample_id=ident, sheet=f"sample-sheet-{sheet+1}.png", **record))
        fig.suptitle("Illustrative samples: PlantVillage (left) / PlantDoc train (right)\n"
                     f"Two per category, seed {seed}; original folder labels in sample-manifest.csv", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, .94))
        fig.savefig(output_dir / f"sample-sheet-{sheet+1}.png", dpi=120)
        plt.close(fig)
    write_csv(output_dir / "sample-manifest.csv", samples)
    print(f"Saved {len(samples)} samples across two sheets; raw images unchanged.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get(
        "PLANT_DISEASE_DATA_DIR", Path.home() / "datasets/cecs551-plant-disease")))
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "outputs/image-audit")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    # Generated files must not be placed in a source dataset folder.
    if output_dir == data_root or data_root in output_dir.parents:
        parser.error("Choose an output directory outside the raw dataset root.")
    rows = inspect(data_root, output_dir)
    picture_sheets(rows, data_root, output_dir, args.seed)


if __name__ == "__main__":
    main()
