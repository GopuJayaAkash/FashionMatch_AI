
from pathlib import Path
import pandas as pd
from PIL import Image

from .config import (
    ROOT_DIR,
    CATALOG_CSV,
    CATALOG_DIR
)

REQUIRED_COLUMNS = [
    "product_id",
    "name",
    "category",
    "color",
    "price",
    "image_path",
    "description"
]


def load_catalog(csv_path=CATALOG_CSV):
    csv_path = Path(csv_path)

    if not csv_path.is_absolute():
        csv_path = ROOT_DIR / csv_path

    if not csv_path.exists():
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df = pd.read_csv(csv_path).fillna("")

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    return df[REQUIRED_COLUMNS]


def resolve_image_path(value):
    p = Path(str(value).strip())

    if p.is_absolute():
        return p

    # Handle paths such as data/catalog_hq/15970.jpg
    # and data/catalog/15970.jpg
    if p.parts and p.parts[0] == "data":
        return (ROOT_DIR / p).resolve()

    # Handle paths relative to the project root.
    if (ROOT_DIR / p).exists():
        return (ROOT_DIR / p).resolve()

    # Handle filenames stored without a folder.
    return (CATALOG_DIR / p).resolve()


def validate_catalog(df):
    errors = []

    if df.empty:
        errors.append("The catalogue is empty.")
        return errors

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")

    if "image_path" in df.columns:
        missing = []

        for _, row in df.iterrows():
            p = resolve_image_path(row["image_path"])

            if not p.is_file():
                missing.append(str(row.get("product_id", "?")))

        if missing:
            errors.append(
                "Image files not found for product IDs: "
                + ", ".join(missing[:20])
            )

    return errors


def create_sample_catalog_template(path=CATALOG_CSV):
    path = Path(path)

    if not path.is_absolute():
        path = ROOT_DIR / path

    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        pd.DataFrame(
            columns=REQUIRED_COLUMNS
        ).to_csv(path, index=False)

