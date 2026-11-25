
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

    if not value or value.lower() == "nan":
        return default

    return value


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
    "metadata matching and attribute-aware ranking"
)


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
# SIDEBAR SEARCH SETTINGS
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

    st.subheader("Optional LLM query enhancement")

    use_ollama = st.checkbox(
        "Use local Ollama",
        value=False
    )

    ollama_model = st.text_input(
        "Ollama model",
        value="llama3.2",
        disabled=not use_ollama
    )

    st.caption(
        "If Ollama is unavailable, the original description "
        "is used where supported by the query generator."
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

    # CLEAR PREVIOUS SEARCH DATA BEFORE VALIDATION
    # This prevents stale recommendations from appearing
    # when the user submits an invalid search.

    for key in [
        "results",
        "enhanced_query",
        "ranking_method",
        "search_mode",
        "selected_top_k",
        "selected_category"
    ]:

        st.session_state.pop(key, None)

    # INPUT VALIDATION

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

                # SELECT RANKING IMPLEMENTATION

                if ranking_method == "Experimental (Attribute-aware)":

                    selected_search_function = search_experimental

                else:

                    selected_search_function = search_baseline

                # PERFORM MULTIMODAL PRODUCT SEARCH

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

                # STORE NEW SEARCH RESULTS

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

            with cols[i % 3]:

                # ------------------------------------------
                # PRODUCT IMAGE
                # ------------------------------------------

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

                # ------------------------------------------
                # PRODUCT INFORMATION
                # ------------------------------------------

                product_name = safe_text(
                    row.get("name")
                )

                product_id = safe_text(
                    row.get("product_id")
                )

                category = safe_text(
                    row.get("category")
                )

                color = safe_text(
                    row.get("color")
                )

                st.markdown(f"**{product_name}**")

                st.write(f"Product ID: {product_id}")
                st.write(f"Category: {category}")
                st.write(f"Colour: {color}")

                # ------------------------------------------
                # RECOMMENDATION SCORES
                # ------------------------------------------

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

                # ------------------------------------------
                # RECOMMENDATION EXPLANATION
                # ------------------------------------------

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

                st.divider()


# --------------------------------------------------
# HOW THE SYSTEM WORKS
# --------------------------------------------------

else:

    st.divider()

    st.subheader("How FashionMatch AI works")

    st.markdown(
        """
        1. Product images and metadata are stored in the catalogue.
        2. CLIP generates image and text embeddings.
        3. FAISS retrieves visually and semantically similar products.
        4. Metadata matching improves product relevance.
        5. The selected ranking algorithm orders the recommendations.
        """
    )

    st.caption(
        "FashionMatch AI is a multimodal fashion product "
        "discovery prototype developed as an AI/ML project."
    )


