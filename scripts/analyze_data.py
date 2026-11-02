"""Inventory the prepared PlantVillage/PlantDoc images without modifying them.

Run from the project root, after scripts/prepare_data.py:
    python scripts/analyze_data.py

Large generated CSV/JSON files stay under ignored outputs/eda/. The source
split lists and leaf-group metadata are used as provided, not rewritten.
"""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps, ImageStat


ROOT = Path.home() / "datasets" / "cecs551-plant-disease"
OUTPUT = Path("outputs") / "eda"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
INVENTORY_COLUMNS = ["dataset", "category", "split", "relative_path", "leaf_id",
                     "bytes", "sha256", "readable", "decode_error", "width",
                     "height", "mode", "edge_variance", "sampled"]
FLAG_COLUMNS = ["flag", "dataset", "category", "split", "relative_path", "details"]


def image_key(value):
    """Normalize a source split entry to category/filename under raw/color/."""
    parts = value.strip().replace("\\", "/").split("/")
    if "raw" in parts:
        offset = parts.index("raw")
        if parts[offset:offset + 2] == ["raw", "color"]:
            parts = parts[offset + 2:]
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Unexpected PlantVillage split path: {value!r}")
    return "/".join(parts)


def read_splits(directory):
    assignments = {}
    for name in ("train", "test"):
        path = directory / "splits" / f"color_{name}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Required PlantVillage split list missing: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            key = image_key(line)
            if key in assignments:
                raise ValueError(f"Repeated or conflicting PlantVillage split entry: {key}")
            assignments[key] = name
    return assignments


def read_leaf_map(directory):
    path = directory / "leaf_grouping" / "leaf-map.json"
    if not path.is_file():
        raise FileNotFoundError(f"Required PlantVillage leaf metadata missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Expected PlantVillage leaf-map.json to be a dictionary")
    return data


def leaf_identifier(filename, category, leaf_map):
    """Follow the source dataset loader's leaf-map lookup and disambiguation.

    If the source doesn't resolve an identifier, leave it empty. Do not equate
    filenames merely because their numeric stems happen to match.
    """
    identifier = filename.replace("_final_masked", "")
    if "___" in identifier:
        identifier = identifier.split("___")[-1]
    identifier = identifier.split("copy")[0]
    identifier = Path(identifier).stem.lower().strip()
    suggestions = leaf_map.get(identifier, [])
    if isinstance(suggestions, str):
        suggestions = [suggestions]
    if not isinstance(suggestions, list):
        return ""
    if len(suggestions) == 1:
        return str(suggestions[0])
    matches = [str(item) for item in suggestions if category in str(item)]
    return matches[0] if matches else ""


def image_files(directory):
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset image directory missing: {directory}")
    return sorted(path for path in directory.rglob("*")
                  if path.is_file() and not path.is_symlink()
                  and path.suffix.lower() in EXTENSIONS)


def source_images(root, split_map, leaf_map):
    """Yield paths with the dataset's original category and split labels."""
    for path in image_files(root / "plantvillage" / "raw" / "color"):
        key = path.relative_to(root / "plantvillage" / "raw" / "color").as_posix()
        if len(Path(key).parts) != 2:
            raise ValueError(f"Unexpected PlantVillage category structure: {path}")
        yield path, "plantvillage", path.parent.name, split_map.get(key, "unassigned"), leaf_identifier(path.name, path.parent.name, leaf_map)
    for split in ("train", "test"):
        for path in image_files(root / "plantdoc" / split):
            key = path.relative_to(root / "plantdoc" / split)
            if len(key.parts) != 2:
                raise ValueError(f"Unexpected PlantDoc category structure: {path}")
            yield path, "plantdoc", path.parent.name, split, ""


def edge_variance(image):
    """A reproducible sharpness *proxy*; low values need visual review."""
    thumbnail = ImageOps.exif_transpose(image).convert("L")
    thumbnail.thumbnail((256, 256))
    return round(ImageStat.Stat(thumbnail.filter(ImageFilter.FIND_EDGES)).var[0], 3)


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in columns} for row in rows)


def analyze(data_root, output_dir, samples_per_category=4, seed=551):
    """Scan images, report evidence and flag *candidates* for manual review."""
    if samples_per_category < 1:
        raise ValueError("--samples-per-category must be positive")
    pv_dir = data_root / "plantvillage"
    split_map = read_splits(pv_dir)
    leaf_map = read_leaf_map(pv_dir)
    sources = list(source_images(data_root, split_map, leaf_map))
    if not sources:
        raise ValueError(f"No supported images were found under {data_root}")

    by_class = defaultdict(list)
    for path, dataset, category, _, _ in sources:
        by_class[(dataset, category)].append(path)
    rng = random.Random(seed)
    sampled_paths = set()
    for paths in by_class.values():
        sampled_paths.update(rng.sample(paths, min(samples_per_category, len(paths))))

    inventory = []
    flags = []
    hashes = defaultdict(list)
    groups = defaultdict(list)
    seen_split_entries = set()
    for index, (path, dataset, category, split, leaf_id) in enumerate(sources, 1):
        relative_path = path.relative_to(data_root).as_posix()
        if dataset == "plantvillage":
            seen_split_entries.add(path.relative_to(pv_dir / "raw" / "color").as_posix())
        record = {"dataset": dataset, "category": category, "split": split,
                  "relative_path": relative_path, "leaf_id": leaf_id,
                  "bytes": path.stat().st_size, "sha256": "", "readable": False,
                  "decode_error": "", "width": "", "height": "", "mode": "",
                  "edge_variance": "", "sampled": path in sampled_paths}
        # A byte-wise hash detects exact copies, not re-encoded or cropped variants.
        with path.open("rb") as stream:
            record["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        try:
            with Image.open(path) as image:
                image.load()
                record.update(readable=True, width=image.width, height=image.height, mode=image.mode)
                if path in sampled_paths:
                    record["edge_variance"] = edge_variance(image)
        except (OSError, ValueError, SyntaxError) as error:
            record["decode_error"] = f"{type(error).__name__}: {error}"
            flags.append({**record, "flag": "unreadable", "details": record["decode_error"]})
        if split == "unassigned":
            flags.append({**record, "flag": "unassigned_split", "details": "Absent from both source split lists"})
        inventory.append(record)
        hashes[record["sha256"]].append(record)
        if leaf_id:
            groups[(dataset, leaf_id)].append(record)
        if index % 10000 == 0:
            print(f"Inspected {index:,}/{len(sources):,} images", flush=True)

    missing_paths = sorted(set(split_map) - seen_split_entries)
    for key in missing_paths:
        flags.append({"flag": "missing_split_file", "dataset": "plantvillage",
                      "category": key.split("/")[0], "split": split_map[key],
                      "relative_path": f"plantvillage/raw/color/{key}",
                      "details": "Listed in source split but image not found"})
    duplicate_groups = 0
    cross_split_hash_groups = 0
    for digest, records in hashes.items():
        if len(records) < 2:
            continue
        duplicate_groups += 1
        cross_split = len({(r["dataset"], r["split"]) for r in records}) > 1
        cross_split_hash_groups += int(cross_split)
        for record in records:
            flags.append({**record, "flag": "exact_duplicate",
                          "details": f"SHA-256 {digest}; {len(records)} copies; cross-partition={cross_split}"})
    cross_split_leaf_groups = 0
    for (dataset, leaf_id), records in groups.items():
        if len({r["split"] for r in records}) < 2:
            continue
        cross_split_leaf_groups += 1
        for record in records:
            flags.append({**record, "flag": "same_leaf_across_splits",
                          "details": f"{dataset} leaf_id={leaf_id}; splits={','.join(sorted({r['split'] for r in records}))}"})

    counts = Counter((r["dataset"], r["category"], r["split"]) for r in inventory)
    readable_counts = Counter((r["dataset"], r["category"], r["split"])
                              for r in inventory if r["readable"])
    class_rows = [{"dataset": dataset, "category": category, "split": split,
                   "file_count": count, "readable_count": readable_counts[(dataset, category, split)],
                   "unreadable_count": count - readable_counts[(dataset, category, split)]}
                  for (dataset, category, split), count in sorted(counts.items())]
    samples = [r for r in inventory if r["sampled"]]
    dimensions = [{key: r[key] for key in ("dataset", "category", "split", "relative_path", "width", "height", "mode")}
                  for r in inventory if r["readable"]]
    manifest_path = data_root / "download-manifest.json"
    metadata = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    summary = {"analyzed_utc": datetime.now(timezone.utc).isoformat(),
               "data_root": str(data_root),
               "plantvillage_revision": metadata.get("plantvillage_revision", "unknown"),
               "plantdoc_revision": metadata.get("plantdoc_revision", "unknown"),
               "images_by_dataset": dict(sorted(Counter(r["dataset"] for r in inventory).items())),
               "unreadable_images": sum(not r["readable"] for r in inventory),
               "unassigned_source_images": sum(r["split"] == "unassigned" for r in inventory),
               "missing_source_images": len(missing_paths),
               "resolved_plantvillage_leaf_ids": sum(r["dataset"] == "plantvillage" and bool(r["leaf_id"]) for r in inventory),
               "exact_duplicate_hash_groups": duplicate_groups,
               "exact_duplicate_cross_partition_groups": cross_split_hash_groups,
               "same_leaf_cross_split_groups": cross_split_leaf_groups,
               "sample": {"selection": "seeded uniform random sample per dataset/category, ignoring split",
                          "seed": seed, "requested_per_category": samples_per_category,
                          "selected_images": len(samples),
                          "sharpness": "edge variance on resized grayscale image; relative proxy, not verified blur"}}

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "image_inventory.csv", INVENTORY_COLUMNS, inventory)
    write_csv(output_dir / "class_counts.csv", ["dataset", "category", "split", "file_count", "readable_count", "unreadable_count"], class_rows)
    write_csv(output_dir / "image_dimensions.csv", ["dataset", "category", "split", "relative_path", "width", "height", "mode"], dimensions)
    write_csv(output_dir / "sample_manifest.csv", INVENTORY_COLUMNS, samples)
    write_csv(output_dir / "quality_flags.csv", FLAG_COLUMNS, flags)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--samples-per-category", type=int, default=4)
    parser.add_argument("--seed", type=int, default=551)
    args = parser.parse_args()
    analyze(args.data_root.expanduser(), args.output_dir, args.samples_per_category, args.seed)


if __name__ == "__main__":
    main()
