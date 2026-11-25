
import json
import re

import numpy as np
import pandas as pd
import faiss

from .config import INDEX_FILE, MAPPING_FILE
from .catalog import load_catalog
from .clip_engine import CLIPEncoder


# --------------------------------------------------
# RANKING CONFIGURATION
# --------------------------------------------------

METADATA_BOOST = 0.06
ATTRIBUTE_BOOST = 0.12

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

    # Colors
    "gray": "grey",
    "navy": "blue",
    "steel": "silver",

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
    "striped": "stripe",
}


# Original token-level weights
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


# Experimental attribute group weights
ATTRIBUTE_TYPE_WEIGHTS = {
    "color": 0.35,
    "category": 0.25,
    "gender": 0.15,
    "design": 0.25,
}


# Recognized attributes
COLOR_ATTRIBUTES = {
    "black", "white", "red", "blue", "green",
    "yellow", "pink", "purple", "brown", "grey",
    "beige", "silver", "gold"
}

CATEGORY_ATTRIBUTES = {
    "shirt", "tshirt", "trouser", "shoe", "watch"
}

GENDER_ATTRIBUTES = {
    "men", "women"
}


# --------------------------------------------------
# GRADED DESIGN MATCHING
# --------------------------------------------------

DESIGN_GROUPS = {
    "printed": "printed",
    "graphic": "graphic",
    "stripe": "striped",
    "checked": "checked",
    "check": "checked",
    "solid": "solid",
}

DESIGN_MATCH_WEIGHTS = {
    "printed": {
        "printed": 1.00,
        "graphic": 0.75,
        "striped": 0.50,
        "checked": 0.50,
        "solid": 0.00,
    },

    "graphic": {
        "graphic": 1.00,
        "printed": 0.75,
        "striped": 0.50,
        "checked": 0.50,
        "solid": 0.00,
    },

    "striped": {
        "striped": 1.00,
        "checked": 0.50,
        "printed": 0.50,
        "graphic": 0.50,
        "solid": 0.00,
    },

    "checked": {
        "checked": 1.00,
        "striped": 0.50,
        "printed": 0.50,
        "graphic": 0.50,
        "solid": 0.00,
    },

    "solid": {
        "solid": 1.00,
        "printed": 0.00,
        "graphic": 0.00,
        "striped": 0.00,
        "checked": 0.00,
    },
}

DESIGN_ATTRIBUTES = set(DESIGN_GROUPS.values())


# --------------------------------------------------
# TEXT PROCESSING
# --------------------------------------------------

def normalize_tokens(value):

    text = str(value).lower()

    # Normalize T-shirt variations
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
# ORIGINAL METADATA RELEVANCE SCORE
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
# ATTRIBUTE EXTRACTION
# --------------------------------------------------

def extract_attributes(value):

    tokens = normalize_tokens(value)

    normalized_designs = {
        DESIGN_GROUPS[token]
        for token in tokens
        if token in DESIGN_GROUPS
    }

    return {
        "color": tokens & COLOR_ATTRIBUTES,
        "category": tokens & CATEGORY_ATTRIBUTES,
        "gender": tokens & GENDER_ATTRIBUTES,
        "design": normalized_designs,
    }


# --------------------------------------------------
# ATTRIBUTE-AWARE RELEVANCE SCORE
# --------------------------------------------------

def calculate_attribute_score(query_text, row):

    if not query_text or not query_text.strip():
        return 0.0

    query_attributes = extract_attributes(query_text)

    # Catalogue color and category are authoritative.
    # Gender comes from the product name.
    # Design comes from product name and description.

    product_attributes = {
        "color": extract_attributes(
            str(row.get("color", ""))
        )["color"],

        "category": extract_attributes(
            str(row.get("category", ""))
        )["category"],

        "gender": extract_attributes(
            str(row.get("name", ""))
        )["gender"],

        "design": extract_attributes(
            " ".join([
                str(row.get("name", "")),
                str(row.get("description", ""))
            ])
        )["design"]
    }

    total_weight = 0.0
    matched_weight = 0.0

    for attribute_type, query_values in query_attributes.items():

        if not query_values:
            continue

        weight = ATTRIBUTE_TYPE_WEIGHTS[attribute_type]

        total_weight += weight

        # Graded matching for design attributes
        if attribute_type == "design":

            product_designs = product_attributes["design"]

            design_scores = []

            for query_design in query_values:

                possible_scores = [
                    DESIGN_MATCH_WEIGHTS
                    .get(query_design, {})
                    .get(product_design, 0.0)
                    for product_design in product_designs
                ]

                best_match = (
                    max(possible_scores)
                    if possible_scores
                    else 0.0
                )

                design_scores.append(best_match)

            match_ratio = (
                sum(design_scores) / len(design_scores)
            )

        else:

            # Exact matching for other attributes
            matched_values = (
                query_values & product_attributes[attribute_type]
            )

            match_ratio = (
                len(matched_values) / len(query_values)
            )

        matched_weight += weight * match_ratio

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
    # EXPERIMENTAL ATTRIBUTE-AWARE RERANKING
    # --------------------------------------------------

    # Preserve original metadata score for comparison
    df["metadata_score"] = df.apply(
        lambda row: calculate_metadata_score(
            query_text,
            row
        ),
        axis=1
    )

    # Calculate graded attribute-aware score
    df["attribute_score"] = df.apply(
        lambda row: calculate_attribute_score(
            query_text,
            row
        ),
        axis=1
    )

    # Experimental ranking formula
    df["ranking_score"] = (
        df["similarity"]
        + ATTRIBUTE_BOOST * df["attribute_score"]
    )

    # Stable ranking
    results = (
        df.sort_values(
            by="ranking_score",
            ascending=False,
            kind="mergesort"
        )
        .head(top_k)
        .reset_index(drop=True)
    )

    return results


