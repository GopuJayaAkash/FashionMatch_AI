
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from fashionmatch.search_experimental import search_products


# --------------------------------------------------
# Q4 INSPECTION
# --------------------------------------------------

evaluation_df = pd.read_csv(
    ROOT_DIR / "data" / "evaluation_queries.csv"
).fillna("")

q4 = evaluation_df[
    evaluation_df["query_id"] == "Q4"
].iloc[0]

query_text = str(q4["query_text"])
category = str(q4["category"])

relevant_ids = set(
    str(q4["expected_relevant_product_ids"]).split(";")
)

relevant_ids.discard("")


# Retrieve all matching products
results = search_products(
    query_text=query_text,
    category=category,
    max_price=None,
    top_k=1500
)

results["is_relevant"] = (
    results["product_id"].astype(str).isin(relevant_ids)
)

# Show the top 20 recommendations
columns = [
    "product_id",
    "name",
    "category",
    "color",
    "similarity",
    "metadata_score",
    "attribute_score",
    "ranking_score",
    "is_relevant"
]

print("\nQUERY:", query_text)
print("CATEGORY:", category)

print("\nTOP 20 EXPERIMENTAL RECOMMENDATIONS")
print("=" * 110)

print(
    results[columns]
    .head(20)
    .to_string(index=False)
)

# Inspect relevant products even if they rank lower
print("\nALL LABELLED RELEVANT PRODUCTS")
print("=" * 110)

relevant_results = results[
    results["is_relevant"]
].sort_values(
    "ranking_score",
    ascending=False
)

print(
    relevant_results[columns]
    .to_string(index=False)
)

# Save complete inspection results
output_path = (
    ROOT_DIR / "data" / "q4_attribute_inspection.csv"
)

results[columns].to_csv(
    output_path,
    index=False
)

print("\nFull inspection saved to:")
print(output_path)
