
from pathlib import Path
import sys

# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd

from fashionmatch.search_experimental import search_products

EVALUATION_FILE = PROJECT_DIR / "data" / "evaluation_queries.csv"
OUTPUT_FILE = PROJECT_DIR / "data" / "evaluation_results_experimental.csv"
TOP_K = 10
K_VALUES = [5, 10]


# --------------------------------------------------
# METRIC CALCULATIONS
# --------------------------------------------------

def calculate_precision(recommended_ids, relevant_ids, k):
    """Calculate Precision@K."""
    top_k = recommended_ids[:k]

    if not top_k:
        return 0.0

    relevant_count = sum(
        product_id in relevant_ids
        for product_id in top_k
    )

    # Fixed denominator: K
    return relevant_count / k


def calculate_recall(recommended_ids, relevant_ids, k):
    """Calculate Recall@K."""
    if not relevant_ids:
        return 0.0

    top_k = recommended_ids[:k]

    relevant_count = sum(
        product_id in relevant_ids
        for product_id in top_k
    )

    return relevant_count / len(relevant_ids)


def calculate_average_precision(recommended_ids, relevant_ids, k=10):
    """Calculate AP@K using the known relevance labels."""
    if not relevant_ids:
        return 0.0

    hits = 0
    precision_sum = 0.0

    for rank, product_id in enumerate(recommended_ids[:k], start=1):
        if product_id in relevant_ids:
            hits += 1
            precision_sum += hits / rank

    denominator = min(len(relevant_ids), k)

    if denominator == 0:
        return 0.0

    return precision_sum / denominator


# --------------------------------------------------
# LOAD EVALUATION QUERIES
# --------------------------------------------------

df = pd.read_csv(
    EVALUATION_FILE,
    dtype=str
).fillna("")

required_columns = [
    "query_id",
    "query_text",
    "category",
    "expected_relevant_product_ids"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing column: {column}")


# --------------------------------------------------
# EVALUATE EACH QUERY
# --------------------------------------------------

results_summary = []

print("\n========== FASHIONMATCH AI EVALUATION ==========\n")
print("Ranking method: Experimental (Attribute-aware)")
print(f"Maximum recommendations evaluated: {TOP_K}")


for _, row in df.iterrows():

    query_id = row["query_id"]
    query_text = row["query_text"]
    category = row["category"]

    relevant_ids = {
        product_id.strip()
        for product_id in row[
            "expected_relevant_product_ids"
        ].split(";")
        if product_id.strip()
    }

    if not relevant_ids:
        print(f"\nSkipping {query_id}: No relevance labels.")
        continue

    recommendations = search_products(
        query_image=None,
        query_text=query_text,
        category=category,
        max_price=None,
        top_k=TOP_K
    )

    recommended_ids = (
        recommendations["product_id"]
        .astype(str)
        .tolist()
    )

    p5 = calculate_precision(
        recommended_ids, relevant_ids, 5
    )

    p10 = calculate_precision(
        recommended_ids, relevant_ids, 10
    )

    r5 = calculate_recall(
        recommended_ids, relevant_ids, 5
    )

    r10 = calculate_recall(
        recommended_ids, relevant_ids, 10
    )

    ap10 = calculate_average_precision(
        recommended_ids, relevant_ids, TOP_K
    )

    results_summary.append({
        "Query ID": query_id,
        "Query": query_text,
        "Relevant Products": len(relevant_ids),
        "Precision@5": p5,
        "Precision@10": p10,
        "Recall@5": r5,
        "Recall@10": r10,
        "AP@10": ap10
    })

    print(f"\n{query_id}: {query_text}")
    print("-" * 55)

    print("Recommended IDs:", recommended_ids)
    print("Known relevant products:", len(relevant_ids))

    print(f"Precision@5:  {p5:.2%}")
    print(f"Precision@10: {p10:.2%}")
    print(f"Recall@5:     {r5:.2%}")
    print(f"Recall@10:    {r10:.2%}")
    print(f"AP@10:        {ap10:.2%}")


# --------------------------------------------------
# OVERALL RESULTS
# --------------------------------------------------

if results_summary:

    results_df = pd.DataFrame(results_summary)

    mean_p5 = results_df["Precision@5"].mean()
    mean_p10 = results_df["Precision@10"].mean()
    mean_r5 = results_df["Recall@5"].mean()
    mean_r10 = results_df["Recall@10"].mean()
    map10 = results_df["AP@10"].mean()

    print("\n========== OVERALL BASELINE RESULTS ==========\n")

    print(results_df.to_string(index=False))

    print("\n========== MEAN EVALUATION METRICS ==========")

    print(f"Mean Precision@5:  {mean_p5:.2%}")
    print(f"Mean Precision@10: {mean_p10:.2%}")
    print(f"Mean Recall@5:      {mean_r5:.2%}")
    print(f"Mean Recall@10:     {mean_r10:.2%}")
    print(f"MAP@10:             {map10:.2%}")

    results_df.to_csv(OUTPUT_FILE, index=False)

    print("\nEvaluation report saved to:")
    print(OUTPUT_FILE)

else:
    print("\nNo queries were evaluated.")
