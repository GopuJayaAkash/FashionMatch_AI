
from pathlib import Path
import sys
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from fashionmatch.search import search_products

EVALUATION_FILE = PROJECT_DIR / "data" / "evaluation_queries.csv"

queries = pd.read_csv(EVALUATION_FILE).fillna("")

for _, query in queries.iterrows():

    query_id = str(query["query_id"])
    query_text = str(query["query_text"])
    category = str(query["category"]).strip() or "All"

    expected_ids = {
        item.strip()
        for item in str(
            query["expected_relevant_product_ids"]
        ).split(";")
        if item.strip()
    }

    print("\n" + "=" * 100)
    print(f"QUERY: {query_id} - {query_text}")
    print(f"CATEGORY: {category}")
    print("=" * 100)

    results = search_products(
        query_image=None,
        query_text=query_text,
        category=category,
        max_price=None,
        top_k=10
    )

    if results.empty:
        print("No products found.")
        continue

    for rank, (_, product) in enumerate(
        results.iterrows(), start=1
    ):

        product_id = str(product["product_id"])

        status = (
            "LABELLED RELEVANT"
            if product_id in expected_ids
            else "NOT LABELLED RELEVANT"
        )

        name = str(product.get("name", ""))
        product_category = str(product.get("category", ""))
        color = str(product.get("color", ""))
        description = str(
            product.get("description", "")
        )[:100]

        similarity = float(product["similarity"])
        metadata_score = float(
            product.get("metadata_score", 0.0)
        )
        ranking_score = float(
            product.get("ranking_score", similarity)
        )

        print(f"\nRank: {rank}")
        print(f"Product ID: {product_id}")
        print(f"Name: {name}")
        print(f"Category: {product_category}")
        print(f"Colour: {color}")
        print(f"Description: {description}")
        print(f"CLIP similarity: {similarity:.4f}")
        print(f"Metadata match: {metadata_score:.2%}")
        print(f"Ranking score: {ranking_score:.4f}")
        print(f"Evaluation label: {status}")
