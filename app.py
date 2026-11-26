
import math
import streamlit as st
from PIL import Image, UnidentifiedImageError

from fashionmatch.catalog import (
    load_catalog,
    resolve_image_path,
    create_sample_catalog_template
)

from fashionmatch.search import search_products as search_baseline

from fashionmatch.search_experimental import (
    search_products as search_experimental
)

from fashionmatch.query_generator import generate_search_description
from fashionmatch.llm_assistant import chat_with_fashion_assistant
from fashionmatch.llm_explainer import explain_product_recommendation
from fashionmatch.config import INDEX_FILE


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def safe_float(value, default=0.0):
    """Convert a value to a valid finite float."""

    try:
        number = float(value)

        if math.isfinite(number):
            return number

    except (TypeError, ValueError):
        pass

    return default


def safe_text(value, default="Not available"):
    """Safely display catalogue values."""

    if value is None:
        return default

    value = str(value).strip()

    if not value or value.lower() in ("nan", "none", "<na>"):
        return default

    return value


def build_catalogue_context(results, limit=10):
    """
    Build factual context from retrieved catalogue products.

    Only include scores that actually exist in the result.
    This prevents baseline recommendations from being
    incorrectly described as having an attribute score of zero.
    """

    if results is None or results.empty:
        return ""

    products = []

    for _, row in results.head(limit).iterrows():

        product_lines = [
            f"Product ID: {safe_text(row.get('product_id'))}",
            f"Name: {safe_text(row.get('name'))}",
            f"Category: {safe_text(row.get('category'))}",
            f"Colour: {safe_text(row.get('color'))}",
            f"Price: {safe_text(row.get('price'))}",
            f"Description: {safe_text(row.get('description'))}",
            f"CLIP similarity: {safe_float(row.get('similarity')):.4f}",
            f"Metadata score: {safe_float(row.get('metadata_score')):.4f}",
        ]

        # Include attribute score only when it exists.
        attribute_score = safe_text(
            row.get("attribute_score"),
            default=""
        )

        if attribute_score:
            product_lines.append(
                f"Attribute score: {safe_float(attribute_score):.4f}"
            )

        product_lines.append(
            f"Ranking score: {safe_float(row.get('ranking_score')):.4f}"
        )

        products.append("\n".join(product_lines))

    return "\n\n".join(products)


def get_ollama_model(model):
    """Return the selected model or the default model."""

    model = str(model).strip() if model else ""

    return model or "llama3.2"


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="FashionMatch AI",
    page_icon="👗",
    layout="wide"
)

st.title("👗 FashionMatch AI")

st.caption(
    "Multimodal fashion product discovery using CLIP embeddings, "
    "FAISS retrieval, attribute-aware ranking and local LLM assistance"
)


# --------------------------------------------------
# INITIALIZE SESSION STATE
# --------------------------------------------------

if "fashion_chat_history" not in st.session_state:
    st.session_state["fashion_chat_history"] = []

if "product_explanations" not in st.session_state:
    st.session_state["product_explanations"] = {}

if "comparison_answer" not in st.session_state:
    st.session_state["comparison_answer"] = ""


# --------------------------------------------------
# LOAD PRODUCT CATALOGUE
# --------------------------------------------------

try:

    create_sample_catalog_template()
    catalog = load_catalog()

    if catalog is None or catalog.empty:
        st.error("The product catalogue is empty.")
        st.stop()

except Exception as exc:

    st.error(f"Unable to load the product catalogue: {exc}")
    st.stop()


# --------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------

with st.sidebar:

    st.header("Search settings")

    categories = ["All"] + sorted([
        str(value)
        for value in catalog["category"].dropna().unique()
        if str(value).strip()
    ])

    selected_category = st.selectbox(
        "Product category",
        categories
    )

    top_k = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=30,
        value=12
    )

    st.divider()

    st.subheader("Ranking method")

    ranking_method = st.selectbox(
        "Select recommendation algorithm",
        [
            "Experimental (Attribute-aware)",
            "Baseline (CLIP + Metadata)"
        ],
        index=0
    )

    if ranking_method == "Experimental (Attribute-aware)":

        st.success(
            "Uses CLIP similarity, metadata matching "
            "and attribute-aware ranking."
        )

        st.caption("Attribute boost: 0.12")

    else:

        st.info(
            "Uses CLIP similarity and metadata matching."
        )

    st.divider()

    st.subheader("Ollama AI settings")

    use_ollama = st.checkbox(
        "Enhance search description using Ollama",
        value=False
    )

    ollama_model = st.text_input(
        "Ollama model",
        value="llama3.2"
    )

    ollama_model = get_ollama_model(ollama_model)

    st.caption(
        "Ollama powers the conversational assistant, "
        "AI explanations and product comparisons."
    )

    st.caption(
        "Search enhancement is optional. "
        "Other AI features can be used independently."
    )


# --------------------------------------------------
# IMAGE AND TEXT INPUT
# --------------------------------------------------

left, right = st.columns(2)

with left:

    st.subheader("1. Upload a fashion image")

    uploaded = st.file_uploader(
        "Choose an outfit image",
        type=["png", "jpg", "jpeg", "webp"]
    )

    query_image = None
    image_error = None

    if uploaded is not None:

        try:

            query_image = Image.open(uploaded).convert("RGB")

            st.image(
                query_image,
                caption="Uploaded fashion image",
                use_container_width=True
            )

        except (UnidentifiedImageError, OSError, ValueError):

            image_error = (
                "The uploaded file is not a valid image. "
                "Please upload another image."
            )

            st.error(image_error)


with right:

    st.subheader("2. Describe the outfit (optional)")

    description = st.text_area(
        "Fashion description",
        placeholder=(
            "Example: charcoal grey polo shirt, "
            "blue straight-fit jeans and white sneakers"
        ),
        height=130
    )

    st.info(
        "Search using an image, a text description, or both."
    )


# --------------------------------------------------
# PRODUCT SEARCH
# --------------------------------------------------

if st.button(
    "🔍 Find matching products",
    type="primary",
    use_container_width=True
):

    # Clear previous search results and explanations.

    for key in [
        "results",
        "enhanced_query",
        "ranking_method",
        "search_mode",
        "selected_top_k",
        "selected_category"
    ]:

        st.session_state.pop(key, None)

    st.session_state["product_explanations"] = {}
    st.session_state["comparison_answer"] = ""

    # Input validation.

    if image_error and not description.strip():

        st.warning(
            "Please upload a valid image or enter a text description."
        )

    elif query_image is None and not description.strip():

        st.warning(
            "Upload an image or enter a description first."
        )

    elif not INDEX_FILE.exists():

        st.error(
            "The product search index has not been built. "
            "Follow the README instructions to index your catalogue."
        )

    else:

        with st.spinner(
            "Understanding your query and searching the catalogue..."
        ):

            try:

                enhanced_description = generate_search_description(
                    description.strip(),
                    use_ollama=use_ollama,
                    model=ollama_model
                )

                enhanced_description = (
                    str(enhanced_description).strip()
                    if enhanced_description
                    else ""
                )

                # Select existing ranking implementation.

                if ranking_method == "Experimental (Attribute-aware)":

                    selected_search_function = search_experimental

                else:

                    selected_search_function = search_baseline

                # Perform existing multimodal product search.
                # Ranking formulas remain unchanged.

                results = selected_search_function(
                    query_image=query_image,
                    query_text=enhanced_description,
                    category=selected_category,
                    max_price=None,
                    top_k=top_k
                )

                if results is None or not hasattr(results, "empty"):

                    raise ValueError(
                        "The search algorithm returned an invalid result."
                    )

                # Store results.

                st.session_state["results"] = results

                st.session_state["enhanced_query"] = (
                    enhanced_description
                )

                st.session_state["ranking_method"] = ranking_method

                st.session_state["search_mode"] = (
                    "Image + Text"
                    if query_image is not None and enhanced_description
                    else "Image only"
                    if query_image is not None
                    else "Text only"
                )

                st.session_state["selected_top_k"] = top_k

                st.session_state["selected_category"] = (
                    selected_category
                )

            except Exception as exc:

                st.error(f"Search failed: {exc}")


# --------------------------------------------------
# DISPLAY RECOMMENDED PRODUCTS
# --------------------------------------------------

if "results" in st.session_state:

    results = st.session_state["results"]

    active_method = st.session_state.get(
        "ranking_method",
        "Unknown"
    )

    st.divider()

    st.subheader("3. Recommended products")

    st.caption(
        f"Ranking method: {active_method}"
    )

    st.caption(
        f"Search mode: {st.session_state.get('search_mode', 'Unknown')}"
        f" | Category: {st.session_state.get('selected_category', 'All')}"
        f" | Requested recommendations: "
        f"{st.session_state.get('selected_top_k', 0)}"
    )

    if st.session_state.get("enhanced_query"):

        st.caption(
            "Search description: "
            + st.session_state["enhanced_query"]
        )

    if results.empty:

        st.warning(
            "No matching products found. "
            "Try another query or select a different category."
        )

    else:

        st.success(
            f"Found {len(results)} recommended products."
        )

        if "Attribute-aware" in active_method:

            st.info(
                "Products are ranked using visual similarity, "
                "metadata matching and attribute-aware matching."
            )

        else:

            st.info(
                "Products are ranked using visual similarity "
                "and metadata matching."
            )

        cols = st.columns(3)

        for i, (_, row) in enumerate(results.iterrows()):

            product_id = safe_text(
                row.get("product_id"),
                default=f"unknown_{i}"
            )

            with cols[i % 3]:

                # PRODUCT IMAGE

                image_value = safe_text(
                    row.get("image_path"),
                    default=""
                )

                if image_value:

                    try:

                        image_path = resolve_image_path(image_value)

                        if image_path.exists():

                            st.image(
                                str(image_path),
                                use_container_width=True
                            )

                        else:

                            st.warning(
                                "Product image unavailable."
                            )

                    except Exception:

                        st.warning(
                            "Unable to display this product image."
                        )

                else:

                    st.warning(
                        "Product image unavailable."
                    )

                # PRODUCT INFORMATION

                product_name = safe_text(
                    row.get("name")
                )

                product_category = safe_text(
                    row.get("category")
                )

                color = safe_text(
                    row.get("color")
                )

                st.markdown(f"**{product_name}**")

                st.write(f"Product ID: {product_id}")
                st.write(f"Category: {product_category}")
                st.write(f"Colour: {color}")

                # RECOMMENDATION SCORES

                similarity = safe_float(
                    row.get("similarity")
                )

                metadata_score = safe_float(
                    row.get("metadata_score")
                )

                ranking_score = safe_float(
                    row.get("ranking_score", similarity),
                    default=similarity
                )

                with st.expander("View recommendation scores"):

                    st.metric(
                        "CLIP similarity",
                        f"{similarity:.3f}"
                    )

                    st.caption("CLIP similarity visualization")

                    clip_percentage = max(
                        0.0,
                        min(100.0, similarity * 100)
                    )

                    st.progress(
                        int(round(clip_percentage)),
                        text=f"{clip_percentage:.1f}%"
                    )

                    st.metric(
                        "Metadata match",
                        f"{metadata_score:.0%}"
                    )

                    if "attribute_score" in row.index:

                        attribute_score = safe_float(
                            row.get("attribute_score")
                        )

                        st.metric(
                            "Attribute match",
                            f"{attribute_score:.0%}"
                        )

                    st.metric(
                        "Combined ranking score",
                        f"{ranking_score:.3f}"
                    )

                    st.caption(
                        "These values are ranking signals, "
                        "not probabilities or confidence values."
                    )

                # ORIGINAL RECOMMENDATION EXPLANATION

                with st.expander("Why was this product recommended?"):

                    st.write(
                        "This product was retrieved and ranked "
                        "based on its similarity to your search."
                    )

                    st.write(
                        f"• Visual similarity score: {similarity:.3f}"
                    )

                    st.write(
                        f"• Metadata matching score: {metadata_score:.0%}"
                    )

                    if "attribute_score" in row.index:

                        st.write(
                            f"• Attribute matching score: "
                            f"{safe_float(row.get('attribute_score')):.0%}"
                        )

                    st.caption(
                        "The experimental algorithm additionally "
                        "uses extracted product attributes when ranking."
                        if "Attribute-aware" in active_method
                        else
                        "The baseline algorithm combines CLIP similarity "
                        "with metadata matching."
                    )

                # AI-GENERATED RECOMMENDATION EXPLANATION

                with st.expander("✨ Explain with AI"):

                    explanation_key = f"{product_id}_{i}"

                    if st.button(
                        "Generate AI explanation",
                        key=f"ai_explain_{product_id}_{i}",
                        use_container_width=True
                    ):

                        with st.spinner(
                            "Llama 3.2 is explaining this recommendation..."
                        ):

                            try:

                                explanation = explain_product_recommendation(
                                    product=row,
                                    query=st.session_state.get(
                                        "enhanced_query", ""
                                    ),
                                    ranking_method=active_method,
                                    model=ollama_model
                                )

                                st.session_state[
                                    "product_explanations"
                                ][explanation_key] = explanation

                            except Exception as exc:

                                st.session_state[
                                    "product_explanations"
                                ][explanation_key] = (
                                    f"Unable to generate the AI explanation: {exc}"
                                )

                    if explanation_key in st.session_state[
                        "product_explanations"
                    ]:

                        st.write(
                            st.session_state[
                                "product_explanations"
                            ][explanation_key]
                        )

                st.divider()


# --------------------------------------------------
# PRODUCT COMPARISON
# --------------------------------------------------

if (
    "results" in st.session_state
    and not st.session_state["results"].empty
):

    st.divider()

    st.subheader("4. AI Product Comparison")

    st.write(
        "Select products from your recommendations "
        "to compare them using Llama 3.2."
    )

    comparison_results = st.session_state["results"]

    product_lookup = {}

    for _, row in comparison_results.iterrows():

        product_id = safe_text(row.get("product_id"))

        product_lookup[product_id] = row

    product_ids = list(product_lookup.keys())

    selected_products = st.multiselect(
        "Select products to compare",
        options=product_ids,
        format_func=lambda pid: (
            f"{safe_text(product_lookup[pid].get('name'))} "
            f"(ID: {pid})"
        ),
        max_selections=4,
        key="comparison_product_selection"
    )

    if st.button(
        "⚖️ Compare selected products",
        use_container_width=True
    ):

        if len(selected_products) < 2:

            st.warning(
                "Please select at least two products for comparison."
            )

        else:

            selected_rows = comparison_results[
                comparison_results["product_id"].astype(str).isin(
                    selected_products
                )
            ]

            comparison_context = build_catalogue_context(
                selected_rows,
                limit=4
            )

            comparison_prompt = (
                "Compare the selected fashion products. "
                "Discuss their differences in category, colour, "
                "description, available price and recommendation scores. "
                "Explain which product is more suitable for different "
                "situations based only on the provided information. "
                "Do not invent missing product details. "
                "Do not recalculate or change the original ranking scores."
            )

            with st.spinner(
                "Llama 3.2 is comparing the selected products..."
            ):

                try:

                    comparison_answer = chat_with_fashion_assistant(
                        user_message=comparison_prompt,
                        catalogue_context=comparison_context,
                        model=ollama_model
                    )

                    st.session_state["comparison_answer"] = (
                        comparison_answer
                    )

                except Exception as exc:

                    st.session_state["comparison_answer"] = (
                        f"Unable to compare products: {exc}"
                    )

    if st.session_state.get("comparison_answer"):

        st.markdown("### AI Comparison Result")

        st.write(
            st.session_state["comparison_answer"]
        )


# --------------------------------------------------
# CONVERSATIONAL FASHION AI ASSISTANT
# --------------------------------------------------

st.divider()

st.subheader("3. 💬 Fashion AI Assistant")

st.write(
    "Ask Llama 3.2 about fashion styling, outfit combinations, "
    "occasions, colours or your recommended products."
)

# Prepare context from current recommendations, if available.

current_catalogue_context = ""

if (
    "results" in st.session_state
    and not st.session_state["results"].empty
):

    current_catalogue_context = build_catalogue_context(
        st.session_state["results"],
        limit=10
    )

# Display previous conversation.

for message in st.session_state["fashion_chat_history"]:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# Chat input.

user_message = st.chat_input(
    "Ask FashionMatch AI something..."
)

if user_message:

    # Display and store the new user message.

    st.session_state["fashion_chat_history"].append({
        "role": "user",
        "content": user_message
    })

    with st.chat_message("user"):

        st.markdown(user_message)

    # Use previous conversation as context,
    # excluding the current message.

    conversation_history = (
        st.session_state["fashion_chat_history"][:-1]
    )

    with st.chat_message("assistant"):

        with st.spinner(
            "Fashion AI is thinking..."
        ):

            try:

                assistant_response = chat_with_fashion_assistant(
                    user_message=user_message,
                    conversation_history=conversation_history,
                    catalogue_context=current_catalogue_context,
                    model=ollama_model
                )

            except Exception as exc:

                assistant_response = (
                    f"Unable to contact Ollama: {exc}\n\n"
                    "Please check that Ollama is running and "
                    "the selected model is available."
                )

        st.markdown(assistant_response)

    # Store assistant response.

    st.session_state["fashion_chat_history"].append({
        "role": "assistant",
        "content": assistant_response
    })

    st.rerun()


# Clear conversation history.

if st.session_state["fashion_chat_history"]:

    if st.button(
        "🗑️ Clear chat history",
        key="clear_fashion_chat"
    ):

        st.session_state["fashion_chat_history"] = []

        st.rerun()


# --------------------------------------------------
# HOW THE SYSTEM WORKS
# This section is now ALWAYS visible.
# --------------------------------------------------

st.divider()

st.subheader("How FashionMatch AI works")

st.markdown(
    """
    1. Product images and metadata are stored in the catalogue.
    2. CLIP generates image and text embeddings.
    3. FAISS retrieves visually and semantically similar products.
    4. Metadata and attribute matching improve product relevance.
    5. The selected ranking algorithm orders the recommendations.
    6. Ollama with Llama 3.2 provides conversational assistance,
       AI-powered recommendation explanations and product comparisons.
    """
)

st.caption(
    "FashionMatch AI is a multimodal fashion product "
    "discovery prototype developed as an AI/ML project."
)
