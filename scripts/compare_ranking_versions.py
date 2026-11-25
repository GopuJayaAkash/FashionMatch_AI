
import sys
from pathlib import Path

import pandas as pd

# Add the project root directory to Python's import path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from fashionmatch.search_baseline import (
    search_products as baseline_search
)

from fashionmatch.search_experimental import (
    search_products as experimental_search
)



# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

EVALUATION_FILE = Path("data/evaluation_queries.csv")
OUTPUT_FILE = Path("data/ranking_version_comparison.csv")

TOP_K = 10


# --------------------------------------------------
# EVALUATION FUNCTION
# --------------------------------------------------

def calculate_precision(recommended_ids, relevant_ids, k):

    recommended_top_k = recommended_ids[:k]

    if not recommended_top_k:
        return 0.0

    relevant_count = sum(
        1 for product_id in recommended_top_k
        if product_id in relevant_ids
    )

    return relevant_count / k


# --------------------------------------------------
# MAIN COMPARISON
# --------------------------------------------------

def main():

    evaluation_df = pd.read_csv(
        EVALUATION_FILE
    ).fillna("")

    results = []

    for _, row in evaluation_df.iterrows():

        query_id = row["query_id"]
        query_text = str(row["query_text"])
        category = str(row["category"])

        relevant_ids = set(
            str(row["expected_relevant_product_ids"]).split(";")
        )

        relevant_ids.discard("")

        print("\n" + "=" * 55)
        print(f"Query: {query_id} - {query_text}")
        print("=" * 55)

        # Baseline ranking
        baseline_df = baseline_search(
            query_text=query_text,
            category=category,
            max_price=None,
            top_k=TOP_K
        )

        # Experimental ranking
        experimental_df = experimental_search(
            query_text=query_text,
            category=category,
            max_price=None,
            top_k=TOP_K
        )

        baseline_ids = (
            baseline_df["product_id"]
            .astype(str)
            .tolist()
        )

        experimental_ids = (
            experimental_df["product_id"]
            .astype(str)
            .tolist()
        )

        baseline_p5 = calculate_precision(
            baseline_ids,
            relevant_ids,
            5
        )

        baseline_p10 = calculate_precision(
            baseline_ids,
            relevant_ids,
            10
        )

        experimental_p5 = calculate_precision(
            experimental_ids,
            relevant_ids,
            5
        )

        experimental_p10 = calculate_precision(
            experimental_ids,
            relevant_ids,
            10
        )

        print("\nBaseline Top 10:")
        print(baseline_ids)

        print("\nExperimental Top 10:")
        print(experimental_ids)

        print("\nPrecision Comparison:")

        print(
            f"Baseline P@5:      {baseline_p5:.2%}"
        )

        print(
            f"Experimental P@5:  {experimental_p5:.2%}"
        )

        print(
            f"Baseline P@10:     {baseline_p10:.2%}"
        )

        print(
            f"Experimental P@10: {experimental_p10:.2%}"
        )

        results.append({
            "query_id": query_id,
            "query_text": query_text,

            "baseline_precision_at_5": baseline_p5,
            "experimental_precision_at_5": experimental_p5,

            "baseline_precision_at_10": baseline_p10,
            "experimental_precision_at_10": experimental_p10,

            "baseline_top_10": ";".join(baseline_ids),
            "experimental_top_10": ";".join(experimental_ids)
        })

    # Save individual query results
    results_df = pd.DataFrame(results)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # Calculate average precision
    print("\n" + "=" * 65)
    print("FINAL BASELINE VS EXPERIMENTAL COMPARISON")
    print("=" * 65)

    print(
        "\nBaseline Mean P@5: "
        f"{results_df['baseline_precision_at_5'].mean():.2%}"
    )

    print(
        "Experimental Mean P@5: "
        f"{results_df['experimental_precision_at_5'].mean():.2%}"
    )

    print(
        "\nBaseline Mean P@10: "
        f"{results_df['baseline_precision_at_10'].mean():.2%}"
    )

    print(
        "Experimental Mean P@10: "
        f"{results_df['experimental_precision_at_10'].mean():.2%}"
    )

    print("\nComparison file saved to:")
    print(OUTPUT_FILE.resolve())


if __name__ == "__main__":
    main()
