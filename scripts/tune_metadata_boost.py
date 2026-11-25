
from pathlib import Path
import sys
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from fashionmatch.search import search_products

EVALUATION_FILE = PROJECT_DIR / "data" / "evaluation_queries.csv"
OUTPUT_FILE = PROJECT_DIR / "data" / "metadata_boost_tuning.csv"

BOOST_VALUES = [0.06, 0.08, 0.10, 0.12, 0.15, 0.18]

queries = pd.read_csv(EVALUATION_FILE).fillna("")

all_results = []

for _, query in queries.iterrows():

    query_id = str(query["query_id"])
    query_text = str(query["query_text"])
    category = str(query["category"]).strip() or "All"

    expected_ids = {
        x.strip()
        for x in str(
            query["expected_relevant_product_ids"]
        ).split(";")
        if x.strip()
    }

    # Retrieve candidate products once per query.
    products = search_products(
        query_image=None,
        query_text=query_text,
        category=category,
        max_price=None,
        top_k=1500
    )

    for boost in BOOST_VALUES:

        ranked = products.copy()

        ranked["test_ranking_score"] = (
            ranked["similarity"]
            + boost * ranked["metadata_score"]
        )

        ranked = ranked.sort_values(
            "test_ranking_score",
            ascending=False,
            kind="mergesort"
        ).head(10)

        top5_ids = ranked.head(5)["product_id"].astype(str)
        top10_ids = ranked["product_id"].astype(str)

        p5 = sum(
            product_id in expected_ids
            for product_id in top5_ids
        ) / 5

        p10 = sum(
            product_id in expected_ids
            for product_id in top10_ids
        ) / 10

        all_results.append({
            "query_id": query_id,
            "query": query_text,
            "metadata_boost": boost,
            "precision_at_5": p5,
            "precision_at_10": p10
        })

results_df = pd.DataFrame(all_results)

summary = (
    results_df.groupby("metadata_boost")
    .agg(
        mean_precision_at_5=("precision_at_5", "mean"),
        mean_precision_at_10=("precision_at_10", "mean")
    )
    .reset_index()
)

print("\n========== METADATA BOOST TUNING ==========\n")

print(
    summary.to_string(
        index=False,
        formatters={
            "metadata_boost": "{:.2f}".format,
            "mean_precision_at_5": "{:.2%}".format,
            "mean_precision_at_10": "{:.2%}".format
        }
    )
)

results_df.to_csv(OUTPUT_FILE, index=False)

print(f"\nDetailed results saved to: {OUTPUT_FILE}")
