
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

baseline_file = DATA_DIR / "evaluation_results.csv"
experimental_file = DATA_DIR / "evaluation_results_experimental.csv"

baseline = pd.read_csv(baseline_file)
experimental = pd.read_csv(experimental_file)

metrics = [
    "Precision@5",
    "Precision@10",
    "Recall@5",
    "Recall@10",
    "AP@10"
]

comparison = baseline[["Query ID", "Query"]].copy()

for metric in metrics:
    comparison[f"Baseline {metric}"] = baseline[metric]
    comparison[f"Experimental {metric}"] = experimental[metric]
    comparison[f"Improvement {metric}"] = (
        experimental[metric] - baseline[metric]
    )

comparison_file = DATA_DIR / "final_ranking_comparison.csv"
comparison.to_csv(comparison_file, index=False)

summary = []

for metric in metrics:
    baseline_mean = baseline[metric].mean()
    experimental_mean = experimental[metric].mean()

    summary.append({
        "Metric": metric,
        "Baseline Mean": baseline_mean,
        "Experimental Mean": experimental_mean,
        "Improvement": experimental_mean - baseline_mean
    })

summary_df = pd.DataFrame(summary)

summary_file = DATA_DIR / "final_metrics_summary.csv"
summary_df.to_csv(summary_file, index=False)

print("\n========== FINAL RANKING COMPARISON ==========\n")
print(summary_df.to_string(index=False))

print("\nDetailed comparison saved to:")
print(comparison_file)

print("\nMetric summary saved to:")
print(summary_file)
