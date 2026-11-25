
import os
import shutil
import pandas as pd

CSV_PATH = "data/products.csv"
BACKUP_DIR = "data/catalog"
HQ_DIR = "data/catalog_hq"

df = pd.read_csv(CSV_PATH)

restored = 0
unavailable = []

for index, row in df.iterrows():
    image_path = str(row["image_path"])

    if os.path.isfile(image_path):
        continue

    product_id = str(row["product_id"]).strip()

    # Try original image with the same product ID.
    original_path = os.path.join(
        BACKUP_DIR, f"{product_id}.jpg"
    )

    if os.path.isfile(original_path):
        df.at[index, "image_path"] = original_path
        restored += 1
    else:
        unavailable.append(product_id)

df.to_csv(CSV_PATH, index=False)

print("Restored from original catalogue:", restored)
print("Still unavailable:", len(unavailable))
print("Unavailable product IDs:", unavailable[:30])
print("Catalogue updated.")
