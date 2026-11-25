from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
CATALOG_DIR = DATA_DIR / "catalog"
INDEX_DIR = DATA_DIR / "index"
CATALOG_CSV = DATA_DIR / "products.csv"
INDEX_FILE = INDEX_DIR / "fashion_clip.index"
MAPPING_FILE = INDEX_DIR / "product_ids.json"

# A smaller checkpoint is suitable for an initial laptop prototype.
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
