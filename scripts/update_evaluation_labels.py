from pathlib import Path
import pandas as pd

file_path = Path("data/evaluation_queries.csv")

df = pd.read_csv(file_path, dtype=str).fillna("")

# Additional products confirmed by catalogue metadata
additions = {
    "Q1": ["12951", "15145"],
    "Q2": ["10259"],
    "Q3": ["39988", "22705"],
    "Q5": ["39375", "44983", "58942"]
}

for query_id, new_ids in additions.items():

    mask = df["query_id"] == query_id

    if not mask.any():
        raise ValueError(f"Query {query_id} not found.")

    current_ids = [
        x.strip()
        for x in df.loc[mask, "expected_relevant_product_ids"].iloc[0].split(";")
        if x.strip()
    ]

    updated_ids = list(dict.fromkeys(current_ids + new_ids))

    df.loc[mask, "expected_relevant_product_ids"] = ";".join(updated_ids)

df.to_csv(file_path, index=False, encoding="utf-8")

print("\nEvaluation labels updated successfully!\n")

for _, row in df.iterrows():
    print(
        row["query_id"],
        row["query_text"],
        "->",
        row["expected_relevant_product_ids"]
    )
