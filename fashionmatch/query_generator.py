import requests

def generate_search_description(user_description, use_ollama=False,
                                 model="llama3.2", base_url="http://localhost:11434"):
    """Optional local LLM enhancement. Falls back to the supplied description."""
    if not user_description.strip():
        return ""
    if not use_ollama:
        return user_description.strip()
    prompt = (
        "Rewrite the following fashion description as a concise, detailed product-search "
        "query. Preserve only visible attributes such as garment type, colour, pattern, "
        "material if stated, fit and style. Do not invent brands or prices.\n\n"
        f"Description: {user_description}"
    )
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=90
        )
        response.raise_for_status()
        return response.json().get("response", user_description).strip()
    except requests.RequestException:
        return user_description.strip()
