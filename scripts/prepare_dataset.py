"""Convert Kaggle Fashion Product Images (Small) into FashionMatch's catalogue.

Expected extracted source layout (the script also searches subfolders):
  .../images/<product_id>.jpg
  .../styles.csv

Usage:
  python -m scripts.prepare_dataset --source data/raw_dataset --limit 1500
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path
import pandas as pd
from PIL import Image

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def find_styles_csv(source: Path) -> Path:
    matches = list(source.rglob("styles.csv"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find styles.csv under {source}. Extract the Kaggle dataset first."
        )
    return matches[0]


def build_image_lookup(source: Path) -> dict[str, Path]:
    # Scan once; repeatedly searching the full 44k-image directory is slow.
    lookup: dict[str, Path] = {}
    for path in source.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            lookup.setdefault(path.stem, path)
    return lookup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/raw_dataset", help="Folder containing extracted styles.csv and images")
    parser.add_argument("--limit", type=int, default=1500, help="Maximum number of products to import")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing copied product images")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    source = Path(args.source)
    if not source.is_absolute():
        source = (Path.cwd() / source).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {source}")

    styles_path = find_styles_csv(source)
    # latin1 is tolerant of occasional non-UTF8 product names in the source file.
    styles = pd.read_csv(styles_path, encoding="latin1", on_bad_lines="skip")
    required = {"id", "productDisplayName", "articleType", "baseColour"}
    missing = required - set(styles.columns)
    if missing:
        raise ValueError(f"styles.csv is missing columns: {sorted(missing)}. Found: {list(styles.columns)}")

    catalog_dir = project_root / "data" / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    image_lookup = build_image_lookup(source)
    rows = []
    seen = set()
    for _, item in styles.iterrows():
        if len(rows) >= args.limit:
            break
        if pd.isna(item["id"]):
            continue
        product_id = str(int(item["id"])) if str(item["id"]).replace(".", "", 1).isdigit() else str(item["id"]).strip()
        if not product_id or product_id in seen:
            continue
        source_image = image_lookup.get(product_id)
        if source_image is None:
            continue
        destination_name = f"{product_id}{source_image.suffix.lower()}"
        destination = catalog_dir / destination_name
        if args.overwrite or not destination.exists():
            try:
                with Image.open(source_image) as im:
                    im.convert("RGB").save(destination, quality=92)
            except Exception as exc:
                print(f"Skipping unreadable image {source_image}: {exc}")
                continue

        name = "" if pd.isna(item["productDisplayName"]) else str(item["productDisplayName"]).strip()
        category = "" if pd.isna(item["articleType"]) else str(item["articleType"]).strip()
        color = "" if pd.isna(item["baseColour"]) else str(item["baseColour"]).strip()
        usage = str(item.get("usage", "")).strip() if not pd.isna(item.get("usage", "")) else ""
        gender = str(item.get("gender", "")).strip() if not pd.isna(item.get("gender", "")) else ""
        description = " ".join(x for x in [gender, color, category, usage] if x and x.lower() != "nan")
        rows.append({
            "product_id": product_id,
            "name": name or f"{color} {category}".strip(),
            "category": category,
            "color": color,
            # The source dataset does not provide reliable retail prices.
            "price": "",
            "image_path": destination_name,
            "description": description,
        })
        seen.add(product_id)

    if not rows:
        raise RuntimeError("No product images were imported. Check that styles.csv and its matching images folder are extracted under --source.")

    output_csv = project_root / "data" / "products.csv"
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Imported {len(rows)} products")
    print(f"Catalogue CSV: {output_csv}")
    print(f"Product images: {catalog_dir}")
    print("Note: prices are blank because this dataset does not provide dependable product prices.")


if __name__ == "__main__":
    main()
