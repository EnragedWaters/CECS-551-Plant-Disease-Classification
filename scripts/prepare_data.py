"""Download and check the datasets and starting weights; do not train a model.

Run with the project Python environment:
    python scripts/prepare_data.py

Files go to ~/datasets/cecs551-plant-disease, outside Git and OneDrive.
This downloads roughly 3 GB and keeps the archives for reproducibility. Only
PlantVillage's color variant is extracted. PlantDoc's train/test folders and
PlantVillage's supplied grouping/split metadata are preserved unchanged.
Checks cover archive integrity and image readability, not label correctness,
duplicate detection, or whether the supplied splits suit our experiment.
"""

import hashlib
import json
import shutil
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
import torch
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


ROOT = Path.home() / "datasets" / "cecs551-plant-disease"
PV_REVISION = "9e97599868962bd0079b8db4b7f1efa9185fa1e7"
PD_REVISION = "5467f6012d78d1c446145d5f582da6096f852ae8"
PV_BASE = f"https://huggingface.co/datasets/mohanty/PlantVillage/resolve/{PV_REVISION}"
PV_SHA256 = "fba30c6a7965e49be94b47a62f8aff6cfb1c35c27f475f22092b56db41745e84"


def checksum(path):
    """Fingerprint a file so the exact downloaded copy can be identified."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url, destination, expected_hash=None):
    """Keep incomplete downloads separate; reuse and fingerprint complete files."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        print(f"Downloading {destination.name}", flush=True)
        partial = destination.with_name(destination.name + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "plant-disease-course-project"})
        with urllib.request.urlopen(request, timeout=120) as source, partial.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        if expected_hash and checksum(partial) != expected_hash:
            raise RuntimeError(f"Checksum mismatch: {partial}")
        partial.rename(destination)
    digest = checksum(destination)
    if expected_hash and digest != expected_hash:
        raise RuntimeError(f"Checksum mismatch: {destination}")
    print(f"Ready: {destination.name} ({destination.stat().st_size:,} bytes)", flush=True)
    return {"url": url, "path": str(destination.relative_to(ROOT)),
            "bytes": destination.stat().st_size, "sha256": digest}


def extract(archive, destination, color_only=False):
    """Extract selected files, retaining source paths and checking ZIP checksums."""
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            if member.is_dir():
                continue
            path = Path(member.filename)
            # Reject paths that could write outside the chosen destination.
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError(f"Unsafe archive path: {member.filename}")
            if color_only:
                if path.parts[:2] != ("raw", "color"):
                    continue
                relative = path
            else:
                # GitHub archives contain an enclosing repository-version folder.
                relative = Path(*path.parts[1:])
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            count += 1
    if not count:
        raise RuntimeError(f"No files extracted from {archive}")
    print(f"Extracted {count:,} files into {destination.name}", flush=True)
    return count


def inspect_images(directory):
    """Decode every image without modifying it; record failures for later review."""
    paths = sorted(p for p in directory.rglob("*")
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    failures = []
    total_bytes = 0
    for index, path in enumerate(paths, 1):
        total_bytes += path.stat().st_size
        try:
            with Image.open(path) as image:
                image.load()
        except Exception as error:
            failures.append({"path": str(path.relative_to(directory)), "error": str(error)})
        if index % 10000 == 0:
            print(f"Checked {index:,}/{len(paths):,} images in {directory.name}", flush=True)
    return {"image_files": len(paths), "image_bytes": total_bytes,
            "readable_images": len(paths) - len(failures), "decode_failures": failures}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    pv_archive = ROOT / "archives" / "plantvillage.zip"
    pd_archive = ROOT / "archives" / "plantdoc.zip"
    # The two archives are independent, so download them concurrently.
    with ThreadPoolExecutor(max_workers=2) as pool:
        pv = pool.submit(download, f"{PV_BASE}/data.zip", pv_archive, PV_SHA256)
        pd = pool.submit(download, f"https://codeload.github.com/pratikkayal/PlantDoc-Dataset/zip/{PD_REVISION}", pd_archive)
        sources = [pv.result(), pd.result()]

    # Keep the original documentation and same-leaf/split records with the data.
    for name in ("README.md", "leaf_grouping/leaf-map.json", "splits/color_train.txt", "splits/color_test.txt"):
        sources.append(download(f"{PV_BASE}/{name}", ROOT / "plantvillage" / name))
    extract(pv_archive, ROOT / "plantvillage", color_only=True)
    extract(pd_archive, ROOT / "plantdoc")
    checks = {"plantvillage_color": inspect_images(ROOT / "plantvillage" / "raw" / "color"),
              "plantdoc_train": inspect_images(ROOT / "plantdoc" / "train"),
              "plantdoc_test": inspect_images(ROOT / "plantdoc" / "test")}

    # Download only the explicitly chosen pretrained weights and check the hash
    # prefix published in their filename. weights_only=True restricts loading.
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    weight_path = ROOT / "weights" / Path(weights.url).name
    weight_record = download(weights.url, weight_path)
    expected_prefix = weight_path.stem.rsplit("-", 1)[-1]
    if not weight_record["sha256"].startswith(expected_prefix):
        raise RuntimeError("Pretrained weight checksum mismatch")
    model = mobilenet_v3_small(weights=None)
    model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
    sources.append(weight_record)
    report = {"checked_utc": datetime.now(timezone.utc).isoformat(),
              "plantvillage_revision": PV_REVISION, "plantdoc_revision": PD_REVISION,
              "sources": sources, "image_checks": checks,
              "weights_loaded": True, "training_performed": False}
    (ROOT / "download-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
