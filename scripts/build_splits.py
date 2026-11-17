"""Build repeatable experiment image lists from the full image audit.

Run from the project root, after scripts/analyze_data.py:
    python scripts/build_splits.py

Outputs are CSV manifests under ignored outputs/splits/. Raw images are never
copied, changed, or deleted. This groups byte-identical files and known leaf
IDs; it cannot establish that every differently encoded/cropped image is unique.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


SPLITS = ("train", "validation", "source_test", "target_test")
COLUMNS = ["relative_path", "dataset", "source_category", "source_split",
           "class_id", "class_name", "sha256", "leaf_id", "group_id", "split"]


def fingerprint(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def category_lookup(config):
    """Use one shared, explicit output order for both model versions/datasets."""
    classes = config["classes"]
    if not classes or [c["id"] for c in classes] != list(range(len(classes))):
        raise ValueError("Class IDs must be consecutive, starting at zero")
    lookup = {}
    for item in classes:
        for dataset in ("plantvillage", "plantdoc"):
            key = (dataset, item[dataset])
            if key in lookup:
                raise ValueError(f"Category mapped more than once: {key}")
            lookup[key] = item
    return lookup


def related_groups(rows):
    """Join all evidence transitively, including excluded categories/PD train.

    If A shares a hash with B and B shares a leaf ID with C, all three must
    remain together. Looking at only eligible classes would miss conflicting
    labels attached to an identical image in an excluded category.
    """
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        keys = [("bytes", row["sha256"])]
        if row["leaf_id"]:
            keys.append(("leaf", row["dataset"], row["leaf_id"]))
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    return list(groups.values())


def validate_inventory(rows):
    """Fail on incomplete/unsafe inventory entries rather than guessing splits."""
    seen = set()
    for row in rows:
        path = PurePosixPath(row["relative_path"])
        if path.is_absolute() or ".." in path.parts or "\\" in str(path):
            raise ValueError(f"Unsafe image path: {path}")
        if str(path) in seen:
            raise ValueError(f"Repeated inventory path: {path}")
        seen.add(str(path))
        digest = row["sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Missing/invalid image fingerprint: {path}")
        if row["dataset"] not in ("plantvillage", "plantdoc"):
            raise ValueError(f"Unknown dataset: {row['dataset']}")
        if row["split"] not in ("train", "test", "unassigned"):
            raise ValueError(f"Unknown source split: {row['split']}")
        if str(row["readable"]).lower() not in ("true", "false"):
            raise ValueError(f"Unknown readability flag: {path}")
    if not rows:
        raise ValueError("Empty inventory")


def choose_validation(groups, fraction, seed, class_id):
    """Choose whole groups, approximately 20% of each class's source pool.

    A stable seeded ordering breaks ties. Greedy selection approaches the image
    target while retaining at least one training group. Exact proportions are
    not guaranteed when a physical leaf has several images.
    """
    if len(groups) < 2:
        raise ValueError(f"Class {class_id} needs at least two independent training groups")
    ordered = sorted(groups, key=lambda g: hashlib.sha256(
        f"{seed}:{class_id}:{g[0]['group_id']}".encode()).hexdigest())
    target = sum(len(g) for g in groups) * fraction
    picked = set()
    count = 0
    for group in ordered:
        if len(picked) == len(groups) - 1:
            break
        if abs(count + len(group) - target) < abs(count - target):
            picked.add(group[0]["group_id"])
            count += len(group)
    if not picked:
        group = min(ordered, key=lambda g: abs(len(g) - target))
        picked.add(group[0]["group_id"])
    return picked


def build_manifests(rows, config, seed=551, validation_fraction=0.2,
                    include_plantdoc_train=False):
    """Exclude known risks, preserve holdouts, then split the training pool."""
    if not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    validate_inventory(rows)
    lookup = category_lookup(config)
    manifests = {name: [] for name in SPLITS}
    excluded = []
    pools = defaultdict(list)
    # External test has priority if a connected group also occurs in source
    # data. Source test has priority over source training. Never promote a
    # training image into a test set: only retain its existing test members.
    priorities = {("plantdoc", "test"): 0,
                  ("plantvillage", "test"): 1,
                  ("plantvillage", "train"): 2,
                  ("plantdoc", "train"): 2}

    destinations = {0: "target_test", 1: "source_test", 2: "pool"}

    for group in related_groups(sorted(rows, key=lambda r: r["relative_path"])):
        paths = sorted(r["relative_path"] for r in group)
        group_id = hashlib.sha256("\n".join(paths).encode()).hexdigest()
        labels = defaultdict(set)
        mapped = set()
        for row in group:
            labels[row["dataset"]].add(row["category"])
            item = lookup.get((row["dataset"], row["category"]))
            if item:
                mapped.add(item["id"])
        # Conflicting folder labels are grounds for conservative exclusion,
        # not a diagnosis or permission to choose the nicer-looking label.
        conflict = any(len(values) > 1 for values in labels.values()) or len(mapped) > 1
        eligible = []
        for row in group:
            item = lookup.get((row["dataset"], row["category"]))
            record = {"relative_path": row["relative_path"],
                      "dataset": row["dataset"], "source_category": row["category"],
                      "source_split": row["split"], "sha256": row["sha256"],
                      "leaf_id": row["leaf_id"], "group_id": group_id, "split": "",
                      "class_id": item["id"] if item else "",
                      "class_name": item["plantvillage"] if item else ""}
            reason = None
            if str(row["readable"]).lower() != "true":
                reason = "unreadable"
            elif conflict:
                reason = "conflicting_labels_in_related_group"
            elif item is None:
                reason = "category_outside_primary_comparison"
            elif ((row["dataset"], row["split"]) == ("plantdoc", "train")
                and not include_plantdoc_train):
                reason = "target_training_not_used"
            elif row["split"] == "unassigned":
                reason = "missing_source_split"
            if reason:
                excluded.append({**record, "reason": reason})
            else:
                eligible.append(record)
        if not eligible:
            continue
        priority = min(priorities[(r["dataset"], r["source_split"])] for r in eligible)
        retained, seen_hashes = [], set()
        for record in eligible:
            if priorities[(record["dataset"], record["source_split"])] != priority:
                excluded.append({**record, "reason": "related_to_reserved_test_group"})
            elif record["sha256"] in seen_hashes:
                excluded.append({**record, "reason": "byte_identical_copy"})
            else:
                retained.append(record)
                seen_hashes.add(record["sha256"])
        destination = destinations[priority]
        if destination == "pool":
            pools[retained[0]["class_id"]].append(retained)
        else:
            for record in retained:
                record["split"] = destination
                manifests[destination].append(record)

    for item in config["classes"]:
        groups = pools[item["id"]]
        selected = choose_validation(groups, validation_fraction, seed, item["id"])
        for group in groups:
            name = "validation" if group[0]["group_id"] in selected else "train"
            for record in group:
                record["split"] = name
                manifests[name].append(record)
    for records in manifests.values():
        records.sort(key=lambda r: (r["class_id"], r["relative_path"]))
    excluded.sort(key=lambda r: r["relative_path"])
    verify_manifests(manifests, excluded, rows, len(config["classes"]), include_plantdoc_train)
    return manifests, excluded


def verify_manifests(manifests, excluded, inventory, class_count, include_plantdoc_train=False):
    """Stop if any class lacks coverage or known related images cross splits."""
    all_paths = []
    owner = {}
    for split, records in manifests.items():
        if {r["class_id"] for r in records} != set(range(class_count)):
            raise ValueError(f"Not all selected classes have examples in {split}")
        for row in records:
            all_paths.append(row["relative_path"])
            expected_datasets = {"plantdoc"} if split == "target_test" else {"plantvillage"}
            if include_plantdoc_train and split in ("train", "validation"):
                expected_datasets.add("plantdoc")
            expected_source = "train" if split in ("train", "validation") else "test"
            if row["dataset"] not in expected_datasets or row["source_split"] != expected_source:
                raise ValueError(f"Wrong data source for {split}")
            for key in (("group", row["group_id"]), ("bytes", row["sha256"])):
                if key in owner and owner[key] != split:
                    raise ValueError("Related images cross experiment splits")
                owner[key] = split
    if len({r["sha256"] for records in manifests.values() for r in records}) != len(all_paths):
        raise ValueError("Byte-identical images remain in selected data")
    all_paths.extend(r["relative_path"] for r in excluded)
    if len(set(all_paths)) != len(all_paths) or set(all_paths) != {r["relative_path"] for r in inventory}:
        raise ValueError("Every input image must be selected or excluded exactly once")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=Path("outputs/eda"))
    parser.add_argument("--categories", type=Path, default=Path("configs/categories.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/splits"))
    parser.add_argument("--seed", type=int, default=551)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--include-plantdoc-train", action="store_true")
    args = parser.parse_args()
    if args.include_plantdoc_train and args.output_dir.resolve() == Path("outputs/splits").resolve():
        parser.error("PlantDoc training needs a separate --output-dir; preserve the original splits")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory is not empty; choose a new split directory")
    inventory_path = args.audit_dir / "image_inventory.csv"
    with inventory_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    config = json.loads(args.categories.read_text(encoding="utf-8"))
    audit = json.loads((args.audit_dir / "summary.json").read_text(encoding="utf-8"))
    if dict(Counter(r["dataset"] for r in rows)) != audit["images_by_dataset"]:
        raise ValueError("Audit summary and inventory counts disagree; rerun the audit")
    for key in ("plantvillage_revision", "plantdoc_revision"):
        if not audit.get(key) or audit[key] == "unknown":
            raise ValueError(f"Missing source version {key}; check the download manifest")
    manifests, excluded = build_manifests(
        rows, config, args.seed, args.validation_fraction, args.include_plantdoc_train)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_hashes = {}
    for name, records in {**manifests, "excluded": excluded}.items():
        path = args.output_dir / f"{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS + (["reason"] if name == "excluded" else []))
            writer.writeheader()
            writer.writerows(records)
        output_hashes[path.name] = fingerprint(path)
    per_class = [{"class_id": item["id"], "class_name": item["plantvillage"],
                  **{split: sum(r["class_id"] == item["id"] for r in records)
                     for split, records in manifests.items()}} for item in config["classes"]]
    summary = {"schema_version": 2, "seed": args.seed,
               "include_plantdoc_train": args.include_plantdoc_train,
               "images_by_dataset": {k: dict(sorted(Counter(r["dataset"] for r in v).items()))
                                     for k, v in manifests.items()},
               "validation_fraction_target": args.validation_fraction,
               "inventory_sha256": fingerprint(inventory_path),
               "categories_sha256": fingerprint(args.categories),
               "builder_sha256": fingerprint(Path(__file__)),
               "dataset_revisions": {k: audit[k] for k in ("plantvillage_revision", "plantdoc_revision")},
               "classes": len(config["classes"]), "counts": {k: len(v) for k, v in manifests.items()},
               "known_group_counts": {k: len({r["group_id"] for r in v}) for k, v in manifests.items()},
               "source_images_without_leaf_id": {k: sum(not r["leaf_id"] for r in v)
                                                  for k, v in manifests.items() if k != "target_test"},
               "excluded": len(excluded), "exclusion_reasons": dict(sorted(Counter(r["reason"] for r in excluded).items())),
               "per_class": per_class, "manifest_sha256": output_hashes,
               "limits": "Groups use exact file bytes and resolved leaf metadata only. Missing leaf IDs and re-encoded/cropped duplicates remain a limitation; labels are not independently diagnosed. Counts describe the selected, filtered subset, not all original PlantDoc images."}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"classes": summary["classes"], "counts": summary["counts"],
                      "excluded": summary["excluded"], "exclusion_reasons": summary["exclusion_reasons"]}, indent=2))


if __name__ == "__main__":
    main()
