
import sys
from pathlib import Path

import pandas as pd

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

EVALUATION_FILE = ROOT_DIR / "data" / "evaluation_queries.csv"
OUTPUT_FILE = ROOT_DIR / "data" / "attribute_boost_tuning.csv"

BOOST_VALUES = [0.00, 0.08, 0.12, 0.16, 0.20]

CATALOG_SIZE = 1500
TOP_K = 10


# --------------------------------------------------
# PRECISION CALCULATION
# --------------------------------------------------

def calculate_precision(recommended_ids, relevant_ids, k):

    top_ids = recommended_ids[:k]

    relevant_count = sum(
        1 for product_id in top_ids
        if product_id in relevant_ids
    )

    return relevant_count / k


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    evaluation_df = pd.read_csv(
        EVALUATION_FILE
    ).fillna("")

    all_results = []

    for _, row in evaluation_df.iterrows():

        query_id = str(row["query_id"])
        query_text = str(row["query_text"])
        category = str(row["category"])

        relevant_ids = set(
            str(row["expected_relevant_product_ids"]).split(";")
        )

        relevant_ids.discard("")

        print("\n" + "=" * 60)
        print(f"{query_id}: {query_text}")
        print("=" * 60)

        # Get baseline results
        baseline_df = baseline_search(
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

        baseline_p5 = calculate_precision(
            baseline_ids, relevant_ids, 5
        )

        baseline_p10 = calculate_precision(
            baseline_ids, relevant_ids, 10
        )

        # Retrieve the complete experimental candidate set
        candidates_df = experimental_search(
            query_text=query_text,
            category=category,
            max_price=None,
            top_k=CATALOG_SIZE
        )

        if "attribute_score" not in candidates_df.columns:
            raise ValueError(
                "attribute_score column missing from experimental results."
            )

        # Evaluate each boost independently
        for boost in BOOST_VALUES:

            ranked_df = candidates_df.copy()

            ranked_df["test_ranking_score"] = (
                ranked_df["similarity"]
                + boost * ranked_df["attribute_score"]
            )

            ranked_df = ranked_df.sort_values(
                by="test_ranking_score",
                ascending=False,
                kind="mergesort"
            ).head(TOP_K)

            recommended_ids = (
                ranked_df["product_id"]
                .astype(str)
                .tolist()
            )

            p5 = calculate_precision(
                recommended_ids, relevant_ids, 5
            )

            p10 = calculate_precision(
                recommended_ids, relevant_ids, 10
            )

            all_results.append({
                "query_id": query_id,
                "query_text": query_text,
                "attribute_boost": boost,
                "precision_at_5": p5,
                "precision_at_10": p10,
                "recommended_top_10": ";".join(recommended_ids),
                "baseline_precision_at_5": baseline_p5,
                "baseline_precision_at_10": baseline_p10
            })

            print(
                f"Boost {boost:.2f} | "
                f"P@5: {p5:.2%} | "
                f"P@10: {p10:.2%}"
            )

    # Save detailed results
    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # Calculate average results for each boost
    summary_df = (
        results_df.groupby("attribute_boost")
        .agg(
            mean_precision_at_5=("precision_at_5", "mean"),
            mean_precision_at_10=("precision_at_10", "mean")
        )
        .reset_index()
    )

    print("\n" + "=" * 65)
    print("ATTRIBUTE BOOST TUNING SUMMARY")
    print("=" * 65)

    print(
        summary_df.to_string(
            index=False,
            formatters={
                "mean_precision_at_5": "{:.2%}".format,
                "mean_precision_at_10": "{:.2%}".format
            }
        )
    )

    print("\nBaseline reference:")
    print(
        "Mean P@5: "
        f"{results_df['baseline_precision_at_5'].mean():.2%}"
    )
    print(
        "Mean P@10: "
        f"{results_df['baseline_precision_at_10'].mean():.2%}"
    )

    print("\nDetailed results saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
