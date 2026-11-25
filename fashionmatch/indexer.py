import json
import numpy as np
import faiss
from PIL import Image
from tqdm.auto import tqdm
from .catalog import load_catalog, resolve_image_path, validate_catalog
from .clip_engine import CLIPEncoder
from .config import INDEX_DIR, INDEX_FILE, MAPPING_FILE

def build_index(csv_path=None):
    df = load_catalog(csv_path) if csv_path else load_catalog()
    errors = validate_catalog(df)
    if errors:
        raise ValueError("\\n".join(errors))

    encoder = CLIPEncoder()
    vectors, product_ids = [], []
    for _, row in df.iterrows():
        image_path = resolve_image_path(row["image_path"])
        try:
            image = Image.open(image_path).convert("RGB")
            vector = encoder.encode_images([image])[0]
            vectors.append(vector)
            product_ids.append(str(row["product_id"]))
        except Exception as exc:
            print(f"Skipping {row['product_id']}: {exc}")

    if not vectors:
        raise ValueError("No product images could be encoded.")

    matrix = np.vstack(vectors).astype("float32")
    index = faiss.IndexFlatIP(matrix.shape[1])  # normalized vectors => cosine similarity
    index.add(matrix)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))
    MAPPING_FILE.write_text(json.dumps(product_ids, indent=2), encoding="utf-8")
    return len(product_ids)
