
import os
import io
import shutil
import pandas as pd

from PIL import Image
from datasets import load_dataset

CSV_PATH = "data/products.csv"
OUTPUT_DIR = "data/catalog_hq"
BACKUP_PATH = "data/products_before_hq.csv"

def normalize_id(value):
    try:
        return str(int(float(value)))
    except (ValueError, TypeError):
        return str(value).strip()

def load_product_image(value):
    if isinstance(value, Image.Image):
        return value.convert("RGB")

    if isinstance(value, dict):
        if value.get("bytes"):
            return Image.open(
                io.BytesIO(value["bytes"])
            ).convert("RGB")

        if value.get("path"):
            return Image.open(
                value["path"]
            ).convert("RGB")

    raise ValueError("Unsupported image format")

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(CSV_PATH)

    df = pd.read_csv(CSV_PATH)

    if "product_id" not in df.columns:
        raise ValueError("product_id column not found")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Preserve the original catalogue CSV.
    if not os.path.exists(BACKUP_PATH):
        shutil.copy2(CSV_PATH, BACKUP_PATH)

    wanted_ids = set(
        df["product_id"].map(normalize_id)
    )

    found_ids = set()

    print("Loading higher-resolution fashion dataset...")

    dataset = load_dataset(
        "ceyda/fashion-products-small",
        split="train",
        streaming=True
    )

    for row in dataset:
        product_id = normalize_id(row["id"])

        if product_id not in wanted_ids:
            continue

        if product_id in found_ids:
            continue

        try:
            image = load_product_image(row["image"])

            output_path = os.path.join(
                OUTPUT_DIR,
                f"{product_id}.jpg"
            )

            image.save(
                output_path,
                "JPEG",
                quality=95,
                optimize=True
            )

            found_ids.add(product_id)

            if len(found_ids) % 100 == 0:
                print(
                    f"Downloaded {len(found_ids)} images..."
                )

            if found_ids == wanted_ids:
                break

        except Exception as error:
            print(f"Skipping {product_id}: {error}")

    # Update image paths only for successfully downloaded images.
    def update_path(value):
        product_id = normalize_id(value)

        if product_id in found_ids:
            return f"{OUTPUT_DIR}/{product_id}.jpg"

        return None

    new_paths = df["product_id"].map(update_path)

    if "image_path" not in df.columns:
        raise ValueError("image_path column not found")

    df["image_path"] = new_paths.fillna(df["image_path"])

    df.to_csv(CSV_PATH, index=False)

    print("\nImage replacement completed.")
    print("Original catalogue images preserved.")
    print(f"Matched images: {len(found_ids)}")
    print(f"Unmatched images: {len(wanted_ids - found_ids)}")
    print(f"New image folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
