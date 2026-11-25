
import json
import re

import numpy as np
import pandas as pd
import faiss

from .config import INDEX_FILE, MAPPING_FILE
from .catalog import load_catalog
from .clip_engine import CLIPEncoder


# --------------------------------------------------
# METADATA MATCHING CONFIGURATION
# --------------------------------------------------

# Maximum additional ranking boost from metadata.
# CLIP similarity remains the primary ranking signal.
METADATA_BOOST = 0.06

STOPWORDS = {
    "a", "an", "the", "and", "or", "for", "with",
    "of", "in", "on", "to", "by"
}

TOKEN_ALIASES = {
    # Gender
    "woman": "women",
    "womens": "women",
    "ladies": "women",
    "lady": "women",
    "female": "women",

    "man": "men",
    "mens": "men",
    "male": "men",

    # Colours
    "gray": "grey",
    "navy": "blue",

    # Product types
    "shirts": "shirt",
    "trousers": "trouser",
    "pants": "trouser",
    "shoes": "shoe",
    "sneakers": "shoe",
    "sneaker": "shoe",
    "watches": "watch",

    # T-shirts
    "tshirts": "tshirt",
    "tshirt": "tshirt",
    "tees": "tshirt",
    "tee": "tshirt",

    # Design
    "prints": "printed",
    "print": "printed",
    "graphics": "graphic",
    "stripes": "stripe",
}

ATTRIBUTE_WEIGHTS = {
    "women": 3.0,
    "men": 3.0,

    "black": 3.0,
    "white": 3.0,
    "red": 3.0,
    "blue": 3.0,
    "green": 3.0,
    "yellow": 3.0,
    "pink": 3.0,
    "purple": 3.0,
    "brown": 3.0,
    "grey": 3.0,
    "beige": 3.0,
    "silver": 3.0,
    "gold": 3.0,
    "steel": 3.0,

    "shirt": 2.0,
    "tshirt": 2.0,
    "trouser": 2.0,
    "shoe": 2.0,
    "watch": 2.0,

    "printed": 2.0,
    "graphic": 2.0,
    "stripe": 2.0,
    "checked": 2.0,
    "solid": 2.0,
}


# --------------------------------------------------
# TEXT PROCESSING
# --------------------------------------------------

def normalize_tokens(value):

    text = str(value).lower()

    # Normalize T-shirt variations before tokenization
    text = re.sub(
        r"\bt[\s-]*shirts?\b",
        " tshirt ",
        text
    )

    tokens = re.findall(r"[a-z]+", text)

    normalized = []

    for token in tokens:

        if token in STOPWORDS:
            continue

        token = TOKEN_ALIASES.get(token, token)

        normalized.append(token)

    return set(normalized)


# --------------------------------------------------
# METADATA RELEVANCE SCORE
# --------------------------------------------------

def calculate_metadata_score(query_text, row):

    if not query_text or not query_text.strip():
        return 0.0

    query_tokens = normalize_tokens(query_text)

    if not query_tokens:
        return 0.0

    product_text = " ".join([
        str(row.get("name", "")),
        str(row.get("category", "")),
        str(row.get("color", "")),
        str(row.get("description", ""))
    ])

    product_tokens = normalize_tokens(product_text)

    total_weight = 0.0
    matched_weight = 0.0

    for token in query_tokens:

        weight = ATTRIBUTE_WEIGHTS.get(token, 1.0)

        total_weight += weight

        if token in product_tokens:
            matched_weight += weight

    if total_weight == 0:
        return 0.0

    return matched_weight / total_weight


# --------------------------------------------------
# PRODUCT SEARCH
# --------------------------------------------------

def search_products(
    query_image=None,
    query_text="",
    category="All",
    max_price=None,
    top_k=12,
    image_weight=0.7
):

    if not INDEX_FILE.exists() or not MAPPING_FILE.exists():

        raise FileNotFoundError(
            "Search index not found. Build the catalogue index first."
        )

    index = faiss.read_index(str(INDEX_FILE))

    product_ids = json.loads(
        MAPPING_FILE.read_text(encoding="utf-8")
    )

    df = load_catalog()

    if index.ntotal == 0:
        raise ValueError("The search index is empty.")

    encoder = CLIPEncoder()

    candidates = []
    weights = []

    # Image query
    if query_image is not None:

        image_vector = encoder.encode_images(
            [query_image]
        )[0]

        candidates.append(image_vector)
        weights.append(float(image_weight))

    # Text query
    if query_text.strip():

        text_vector = encoder.encode_texts(
            [query_text.strip()]
        )[0]

        candidates.append(text_vector)

        if query_image is not None:
            weights.append(1.0 - float(image_weight))
        else:
            weights.append(1.0)

    if not candidates:

        raise ValueError(
            "Provide an image, a text description, or both."
        )

    # Normalize query weights
    weights = np.asarray(weights, dtype="float32")
    weights = weights / weights.sum()

    # Combine image and text embeddings
    query = np.average(
        np.vstack(candidates),
        axis=0,
        weights=weights
    ).astype("float32")

    query /= (np.linalg.norm(query) + 1e-12)

    # Retrieve the complete catalogue
    distances, positions = index.search(
        query.reshape(1, -1),
        index.ntotal
    )

    score_by_id = {}

    for score, pos in zip(distances[0], positions[0]):

        if 0 <= pos < len(product_ids):

            product_id = str(product_ids[pos])

            score_by_id[product_id] = float(score)

    # Attach original CLIP scores
    df["product_id"] = df["product_id"].astype(str)

    df["similarity"] = df["product_id"].map(score_by_id)

    df = df.dropna(subset=["similarity"]).copy()

    # Category filter
    if category != "All":

        df = df[
            df["category"]
            .astype(str)
            .str.strip()
            .str.lower()
            == category.strip().lower()
        ].copy()

    # Optional price filter
    if max_price is not None:

        numeric_price = pd.to_numeric(
            df["price"],
            errors="coerce"
        )

        df = df[
            numeric_price <= float(max_price)
        ].copy()

    if df.empty:
        return df

    # --------------------------------------------------
    # METADATA-AWARE RERANKING
    # --------------------------------------------------

    df["metadata_score"] = df.apply(
        lambda row: calculate_metadata_score(
            query_text,
            row
        ),
        axis=1
    )

    # Original CLIP similarity remains unchanged.
    # Metadata provides a small additional ranking boost.
    df["ranking_score"] = (
        df["similarity"]
        + METADATA_BOOST * df["metadata_score"]
    )

    # Rank by combined score
    results = (
        df.sort_values(
            by="ranking_score",
            ascending=False
        )
        .head(top_k)
        .reset_index(drop=True)
    )

    return results
