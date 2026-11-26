
import requests


# --------------------------------------------------
# OLLAMA CONFIGURATION
# --------------------------------------------------

DEFAULT_MODEL = "llama3.2"
DEFAULT_BASE_URL = "http://localhost:11434"


# --------------------------------------------------
# SAFE VALUE CONVERSION
# --------------------------------------------------

def safe_value(value, default="Not available"):
    """Convert catalogue values safely into strings."""

    if value is None:
        return default

    value = str(value).strip()

    if not value or value.lower() in ("nan", "none"):
        return default

    return value


def safe_score(value):
    """Convert a score into a finite float."""

    try:
        number = float(value)

        if number == number and abs(number) != float("inf"):
            return number

    except (TypeError, ValueError):
        pass

    return 0.0


# --------------------------------------------------
# BUILD PRODUCT CONTEXT
# --------------------------------------------------

def build_product_context(product, query="", ranking_method="Unknown"):
    """
    Build factual product information for the LLM.

    Accepts a dictionary or pandas Series.
    """

    if hasattr(product, "to_dict"):
        product = product.to_dict()

    if not isinstance(product, dict):
        raise ValueError("Product information must be a dictionary or Series.")

    context = f"""
CUSTOMER SEARCH
Search query: {safe_value(query)}

RANKING METHOD
{safe_value(ranking_method)}

RECOMMENDED PRODUCT
Product ID: {safe_value(product.get("product_id"))}
Product name: {safe_value(product.get("name"))}
Category: {safe_value(product.get("category"))}
Colour: {safe_value(product.get("color"))}
Price: {safe_value(product.get("price"))}
Description: {safe_value(product.get("description"))}

RECOMMENDATION SCORES
CLIP similarity: {safe_score(product.get("similarity")):.4f}
Metadata match: {safe_score(product.get("metadata_score")):.4f}
Attribute match: {safe_score(product.get("attribute_score")):.4f}
Combined ranking score: {safe_score(product.get("ranking_score")):.4f}
"""

    return context.strip()


# --------------------------------------------------
# AI RECOMMENDATION EXPLANATION
# --------------------------------------------------

def explain_product_recommendation(
    product,
    query="",
    ranking_method="Experimental (Attribute-aware)",
    model=DEFAULT_MODEL,
    base_url=DEFAULT_BASE_URL,
    timeout=120
):
    """
    Generate a natural-language explanation for a
    recommended product using local Ollama.

    The LLM explains existing recommendation results.
    It does not calculate or modify ranking scores.
    """

    try:

        product_context = build_product_context(
            product=product,
            query=query,
            ranking_method=ranking_method
        )

    except (ValueError, TypeError) as exc:
        return f"Unable to prepare product information: {exc}"

    system_prompt = """
You are the recommendation explanation engine for FashionMatch AI.

Explain why a product was recommended based only on the supplied
search query, catalogue information and recommendation scores.

Rules:

1. Explain the relationship between the search query and product.
2. Use the available product attributes, such as category and colour.
3. Explain CLIP similarity as a visual or semantic matching signal.
4. For attribute-aware ranking, discuss the available attribute score.
5. For baseline ranking, discuss metadata matching.
6. Never invent product properties, brands, materials or prices.
7. Never describe a ranking score as a probability or confidence.
8. Do not claim that a product is the perfect match.
9. If information is unavailable, do not guess.
10. Keep the explanation clear and concise.

Important:
You are explaining an existing recommendation.
You must not recalculate, change or reorder product rankings.
"""

    user_prompt = f"""
Explain this FashionMatch AI recommendation.

{product_context}

Write the explanation in approximately 3 to 5 sentences.
"""

    try:

        response = requests.post(
            f"{base_url.rstrip('/')}/api/chat",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0.3
                }
            },
            timeout=timeout
        )

        response.raise_for_status()

        result = response.json()

        explanation = (
            result.get("message", {}).get("content", "")
        )

        if not explanation.strip():
            return "The AI could not generate a recommendation explanation."

        return explanation.strip()

    except requests.exceptions.ConnectionError:

        return (
            "Unable to connect to Ollama. "
            "Please ensure Ollama is running."
        )

    except requests.exceptions.Timeout:

        return (
            "The explanation request timed out. Please try again."
        )

    except requests.exceptions.HTTPError as exc:

        return f"Ollama returned an HTTP error: {exc}"

    except requests.exceptions.RequestException as exc:

        return f"Unable to generate explanation: {exc}"

    except (ValueError, KeyError, TypeError):

        return "Unable to process the Ollama response."
