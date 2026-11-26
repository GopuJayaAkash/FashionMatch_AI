# 👗 FashionMatch AI – Multimodal Fashion Product Recommendation System

An AI-powered fashion product discovery and recommendation system that combines **multimodal search, computer vision, vector similarity retrieval, attribute-aware ranking, and Large Language Model (LLM) assistance**.

FashionMatch AI allows users to search for fashion products using images, text descriptions, or both. It retrieves visually and semantically relevant products and provides AI-generated explanations, product comparisons, and conversational fashion assistance.

---

## 🚀 Key Features

### 1. Multimodal Fashion Search
- Search using a fashion image.
- Search using a natural-language description.
- Combine image and text inputs.
- Retrieve visually and semantically similar products.

### 2. CLIP-Based Feature Extraction
- Uses `openai/clip-vit-base-patch32`.
- Generates image and text embeddings.
- Maps images and text into a shared embedding space.
- Enables cross-modal similarity comparison.

### 3. FAISS-Based Product Retrieval
- Uses FAISS `IndexFlatIP`.
- Performs vector similarity retrieval.
- Uses normalized embeddings and inner-product similarity.
- Retrieves relevant products efficiently.

### 4. Hybrid Recommendation and Ranking

FashionMatch AI implements two recommendation approaches:

**Baseline Ranking**
- CLIP visual and semantic similarity.
- Metadata matching.

Ranking formula:

`Ranking Score = CLIP Similarity + 0.06 × Metadata Score`

**Experimental Attribute-Aware Ranking**
- CLIP similarity.
- Product attribute matching.
- Colour, category, gender-related information, and design attributes.

Ranking formula:

`Ranking Score = CLIP Similarity + 0.12 × Attribute Score`

The experimental approach improves the ranking of products based on relevant fashion attributes.

### 5. Ollama and Llama 3.2 Integration

The application integrates a locally running LLM through Ollama.

Features include:

- **AI Search Enhancement:** Rewrites fashion descriptions into more detailed product-search queries.
- **AI Recommendation Explanation:** Explains why a product was recommended using its catalogue information and ranking signals.
- **AI Product Comparison:** Compares selected products based on available product details and recommendation scores.
- **Conversational Fashion AI Assistant:** Answers fashion-related questions and supports follow-up conversations.

The LLM operates alongside the retrieval system. It does not replace CLIP, FAISS, or the existing ranking algorithms.

### 6. Interactive Streamlit Interface
- Image upload and text search.
- Product category filtering.
- Adjustable number of recommendations.
- Baseline and experimental ranking selection.
- Recommendation score visualization.
- AI-generated explanations.
- Product comparison.
- Conversational fashion assistant.

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| Computer Vision | CLIP |
| Pretrained Model | openai/clip-vit-base-patch32 |
| Deep Learning | PyTorch |
| Model Processing | Hugging Face Transformers |
| Vector Search | FAISS |
| Data Processing | Pandas, NumPy |
| Image Processing | Pillow |
| LLM Runtime | Ollama |
| Language Model | Llama 3.2 |
| Version Control | Git and GitHub |

---

## 🧠 System Architecture

```text
                  USER INPUT
                      |
          +-----------+-----------+
          |                       |
      Fashion Image          Text Description
          |                       |
          v                       v
       CLIP Image             CLIP Text
       Encoder                Encoder
          |                       |
          +-----------+-----------+
                      |
              Shared Embedding
                   Space
                      |
                      v
                FAISS Retrieval
                      |
                      v
             Candidate Products
                      |
             +--------+--------+
             |                 |
             v                 v
       Baseline Ranking   Experimental Ranking
       CLIP + Metadata    Attribute-Aware
             |                 |
             +--------+--------+
                      |
                      v
             Recommended Products
                      |
                      v
              Streamlit Interface
                      |
                      v
                Ollama LLM
                 Llama 3.2
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
       AI Search   Explanation  Comparison
       Enhancement              and Chat
```

---

## 📁 Project Structure

```text
FashionMatch_AI/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── fashionmatch/
│   ├── __init__.py
│   ├── config.py
│   ├── catalog.py
│   ├── clip_engine.py
│   ├── indexer.py
│   ├── query_generator.py
│   ├── search.py
│   ├── search_baseline.py
│   ├── search_experimental.py
│   ├── llm_assistant.py
│   └── llm_explainer.py
│
├── scripts/
│   ├── build_index.py
│   ├── compare_ranking_versions.py
│   ├── create_final_comparison.py
│   ├── evaluate_experimental.py
│   ├── evaluate_recommendations.py
│   ├── fix_missing_images.py
│   ├── inspect_q4_attributes.py
│   ├── inspect_top10.py
│   ├── prepare_dataset.py
│   ├── tune_attribute_boost.py
│   ├── tune_metadata_boost.py
│   ├── update_evaluation_labels.py
│   └── upgrade_images.py
│
├── data/
│   ├── products.csv
│   ├── evaluation_queries.csv
│   ├── evaluation_results.csv
│   ├── evaluation_results_experimental.csv
│   ├── final_metrics_summary.csv
│   ├── final_ranking_comparison.csv
│   └── ranking_version_comparison.csv
│
└── screenshots/
    ├── baseline ranking.png
    ├── experimental ranking.png
    ├── home page.png
    ├── recommendation explanation.png
    └── red printed t-shirt.png
```

Generated model files, virtual environments, product image collections, and FAISS index files are maintained locally and excluded from Git where configured in `.gitignore`.

---

## ⚙️ Installation and Setup

### 1. Clone the Repository

```powershell
git clone https://github.com/GopuJayaAkash/FashionMatch_AI.git

cd FashionMatch_AI
```

### 2. Create a Virtual Environment

```powershell
python -m venv .venv
```

### 3. Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install Python Dependencies

```powershell
python -m pip install -r requirements.txt
```

### 5. Install Ollama

Install Ollama from:

https://ollama.com/

Download the Llama 3.2 model:

```powershell
ollama pull llama3.2
```

Verify the installed model:

```powershell
ollama list
```

Test the model:

```powershell
ollama run llama3.2
```

Ollama must be running locally for the LLM features to work.

### 6. Prepare the Dataset and Build the Search Index

Prepare the product catalogue and image data according to the project dataset setup.

Run:

```powershell
python scripts\prepare_dataset.py
```

Build the FAISS index:

```powershell
python scripts\build_index.py
```

The index is generated locally and used by the application for product retrieval.

### 7. Run FashionMatch AI

From the project root:

```powershell
streamlit run app.py
```

The application will open in your browser.

---

## 🔍 How the Recommendation System Works

### Step 1: User Input

The user uploads an image, enters a fashion description, or supplies both.

### Step 2: Embedding Generation

CLIP converts the image and/or text into embeddings in a shared vector space.

### Step 3: Similarity Retrieval

FAISS retrieves candidate products using vector similarity.

For normalized embeddings, inner-product similarity corresponds to cosine similarity.

### Step 4: Product Ranking

The retrieved products are ranked using either the baseline or experimental approach.

### Step 5: Recommendation Display

The Streamlit application displays the recommended products and their ranking signals.

### Step 6: LLM Assistance

Ollama with Llama 3.2 provides natural-language search enhancement, explanations, comparisons, and conversational fashion advice.

---

## 📊 Evaluation and Results

The recommendation algorithms were evaluated using five fashion search queries:

1. Blue casual shirt.
2. Black trousers.
3. White casual shoes.
4. Red printed T-shirt.
5. Women silver watch.

### Baseline vs Experimental Results

| Evaluation Metric | Baseline | Experimental |
|---|---:|---:|
| Mean Precision@5 | 88.00% | 92.00% |
| Mean Precision@10 | 68.00% | 68.00% |
| Mean Recall@5 | 72.67% | 76.67% |
| Mean Recall@10 | 100.00% | 100.00% |
| MAP@10 | 87.37% | 95.11% |

### Observations

- Experimental ranking improves Mean Precision@5 by 4 percentage points.
- Mean Recall@5 improves by 4 percentage points.
- MAP@10 improves by 7.74 percentage points.
- Both approaches achieve the same Mean Precision@10 and Mean Recall@10 in the evaluated set.

These results are based on five evaluation queries and manually assigned relevance labels. They should not be interpreted as a guarantee of performance on all fashion searches.

**Note:** CLIP similarity, metadata matching, attribute matching, and combined ranking scores are ranking signals, not probabilities or confidence values.

---

## 🧪 Testing

The following application features have been tested:

| Test | Result |
|---|---|
| Text-based product search | Successful |
| Image-based product search | Successful |
| Combined image and text search | Successful |
| Category filtering | Successful |
| Top-K recommendation selection | Successful |
| Baseline ranking | Successful |
| Experimental ranking | Successful |
| Recommendation explanations | Successful |
| AI search enhancement | Successful |
| AI product comparison | Successful |
| Conversational fashion assistant | Successful |
| Follow-up conversation context | Successful |
| Ranking evaluation after LLM integration | Successful |

---

## ⚠️ Limitations

- Recommendation quality depends on the coverage and quality of the product catalogue.
- Missing product metadata can limit attribute matching.
- Product prices may be unavailable for some catalogue entries.
- Evaluation uses a limited set of five manually labelled queries.
- Llama 3.2 generates natural-language responses and may occasionally produce inaccurate fashion advice.
- Local LLM performance depends on available system resources.
- The project is a research and academic prototype rather than a production shopping platform.

---

## 🔮 Future Enhancements

- Expand the fashion product catalogue.
- Improve attribute extraction and classification.
- Introduce additional ranking strategies.
- Extend evaluation to a larger and more diverse query set.
- Add personalized recommendations based on user preferences.
- Improve conversational context management.
- Explore larger or specialized fashion language models.
- Add more advanced multimodal recommendation capabilities.

---

## 👨‍💻 Author

**Gopu Jaya Akash**

M.Tech – KITS Institute of Technology

Academic Year: 2025–2026

---

## 📄 Project Information

FashionMatch AI is an academic AI/ML project developed to explore multimodal retrieval, computer vision, vector similarity search, attribute-aware recommendation algorithms, and local LLM integration.

**Repository:** https://github.com/GopuJayaAkash/FashionMatch_AI
