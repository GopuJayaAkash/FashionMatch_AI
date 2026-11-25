\# FashionMatch AI



\## Multimodal Fashion Product Recommendation Using CLIP and FAISS



FashionMatch AI is an AI-powered fashion product discovery system that recommends visually and semantically similar fashion products using image and text queries.



The project combines CLIP-based multimodal embeddings, FAISS vector retrieval, metadata matching, and an experimental attribute-aware ranking method. A Streamlit interface allows users to search using an uploaded image, a text description, or both.



This project is an M.Tech-level experimental implementation focused on multimodal retrieval and recommendation ranking.



\## 1. Problem Statement



Traditional fashion product search systems often depend on exact keywords, predefined categories, or manually selected filters. These approaches may fail to identify visually similar products when users describe an item using natural language or provide a reference image.



FashionMatch AI addresses this problem through multimodal search. It uses CLIP to represent images and text in a shared embedding space and FAISS to retrieve visually and semantically similar products.



The system further improves recommendation ordering through metadata-aware and attribute-aware ranking.



\## 2. Objectives



\- Develop an image-based fashion product search system.

\- Support natural-language text-based product search.

\- Enable combined image and text queries.

\- Use CLIP embeddings for multimodal similarity.

\- Use FAISS for efficient vector retrieval.

\- Improve ranking using product metadata and extracted attributes.

\- Compare baseline and experimental ranking methods.

\- Evaluate recommendation quality using Precision@K, Recall@K, and Average Precision.



\## 3. Technologies Used



| Technology | Purpose |

|---|---|

| Python | Core programming language |

| PyTorch | Deep learning framework |

| CLIP | Image and text feature extraction |

| Hugging Face Transformers | Loading the pretrained CLIP model |

| FAISS | Vector similarity search |

| NumPy | Numerical operations |

| Pandas | Catalogue and evaluation data processing |

| Streamlit | Interactive web application |

| Pillow | Image loading and processing |

| Scikit-learn | Supporting machine learning utilities |



\*\*Pretrained model:\*\* `openai/clip-vit-base-patch32`



\## 4. Dataset



The project uses the Fashion Products Small dataset, with a working catalogue of approximately 1,500 products.



The catalogue contains product images and associated metadata, including product names, categories, and colors.



The product image collection is maintained locally under the project's data directory.



\*\*Note:\*\* Reliable product prices are not available for the current catalogue. Therefore, price-based filtering is not included in the final application.



\## 5. Project Structure



```text

FashionMatch\_AI/

|

|-- app.py

|-- requirements.txt

|-- README.md

|-- .gitignore

|

|-- fashionmatch/

|   |-- \_\_init\_\_.py

|   |-- config.py

|   |-- catalog.py

|   |-- clip\_engine.py

|   |-- indexer.py

|   |-- query\_generator.py

|   |-- search.py

|   |-- search\_baseline.py

|   |-- search\_experimental.py

|

|-- data/

|   |-- products.csv

|   |-- evaluation\_queries.csv

|   |-- evaluation\_results.csv

|   |-- evaluation\_results\_experimental.csv

|   |-- final\_ranking\_comparison.csv

|   |-- final\_metrics\_summary.csv

|

|-- scripts/

|   |-- prepare\_dataset.py

|   |-- build\_index.py

|   |-- evaluate\_recommendations.py

|   |-- evaluate\_experimental.py

|   |-- create\_final\_comparison.py

|   |-- update\_evaluation\_labels.py

|   |-- tune\_metadata\_boost.py

|   |-- tune\_attribute\_boost.py

|   |-- compare\_ranking\_versions.py

|   |-- inspect\_top10.py

|   |-- inspect\_q4\_attributes.py

|   |-- upgrade\_images.py

|   |-- fix\_missing\_images.py

|

|-- models/

|

|-- .venv/

```



The catalogue images and generated vector index are stored locally and are not necessarily included when sharing the source code.



\## 6. System Architecture



The system consists of the following main stages:



1\. \*\*Input:\*\* The user uploads an image, enters a text description, or provides both.

2\. \*\*Feature extraction:\*\* CLIP converts the available image and text inputs into embeddings.

3\. \*\*Vector retrieval:\*\* FAISS retrieves candidate products using similarity search.

4\. \*\*Metadata matching:\*\* Product metadata is compared with the search description.

5\. \*\*Attribute-aware ranking:\*\* The experimental method additionally evaluates extracted product attributes.

6\. \*\*Recommendation:\*\* Products are ranked and displayed in the Streamlit interface.



\### Architecture Flow



```text

&#x20;      User Input

&#x20;  (Image / Text / Both)

&#x20;           |

&#x20;           v

&#x20;    CLIP Feature Extraction

&#x20;           |

&#x20;           v

&#x20;     FAISS Retrieval

&#x20;           |

&#x20;           v

&#x20;   Candidate Product Results

&#x20;           |

&#x20;           v

&#x20;      Ranking Module

&#x20;       /          \\

&#x20;      /            \\

&#x20;Baseline       Experimental

&#x20;CLIP +         Attribute-aware

&#x20;Metadata       Ranking

&#x20;      \\            /

&#x20;       \\          /

&#x20;           v

&#x20;    Ranked Recommendations

&#x20;           |

&#x20;           v

&#x20;      Streamlit UI

```



\## 7. Ranking Methods



The project implements two ranking methods to compare recommendation quality.



\### 7.1 Baseline: CLIP + Metadata Ranking



The baseline method combines CLIP similarity with metadata matching.



\*\*Ranking formula:\*\*



```text

Baseline Score =

CLIP Similarity + (0.06 x Metadata Score)

```



The metadata score provides an additional ranking boost when product information matches the user's query.



The selected metadata boost is 0.06, based on the project's parameter-tuning experiments.



Implementation: `fashionmatch/search.py`



\### 7.2 Experimental: Attribute-Aware Ranking



The experimental method combines CLIP similarity with an attribute matching score.



It considers product attributes such as:



\- Color

\- Product category

\- Gender-related information

\- Design or pattern



Design matching uses graded similarity between related design descriptions rather than requiring exact keyword matches.



\*\*Ranking formula:\*\*



```text

Experimental Score =

CLIP Similarity + (0.12 x Attribute Score)

```



The selected attribute boost is 0.12, based on the project's tuning experiments.



Implementation: `fashionmatch/search\_experimental.py`



\### 7.3 Ranking Comparison



The application allows the user to select either:



\- Experimental (Attribute-aware)

\- Baseline (CLIP + Metadata)



This makes it possible to compare the recommendations generated by both approaches using the same search input.



\## 8. Evaluation Methodology



The system was evaluated using five manually labelled search queries:



| Query ID | Search Query | Category |

|---|---|---|

| Q1 | Blue casual shirt | Shirts |

| Q2 | Black trousers | Trousers |

| Q3 | White casual shoes | Casual Shoes |

| Q4 | Red printed T-shirt | T-shirts |

| Q5 | Women silver watch | Watches |



The evaluation uses manually assigned relevant product IDs based on catalogue information and relevance judgments.



The following metrics are calculated:



\- \*\*Precision@5:\*\* Proportion of relevant products among the top five recommendations.

\- \*\*Precision@10:\*\* Proportion of relevant products among the top ten recommendations.

\- \*\*Recall@5:\*\* Proportion of labelled relevant products retrieved in the top five.

\- \*\*Recall@10:\*\* Proportion of labelled relevant products retrieved in the top ten.

\- \*\*AP@10:\*\* Average precision of the ranked results up to position ten.

\- \*\*MAP@10:\*\* Mean Average Precision across the five evaluation queries.



\## 9. Experimental Results



The following results were obtained from the current evaluation dataset.



| Evaluation Metric | Baseline | Experimental |

|---|---:|---:|

| Mean Precision@5 | 88.00% | 92.00% |

| Mean Precision@10 | 68.00% | 68.00% |

| Mean Recall@5 | 72.67% | 76.67% |

| Mean Recall@10 | 100.00% | 100.00% |

| MAP@10 | 87.37% | 95.11% |



\### Observations



1\. The experimental method improves Mean Precision@5 from 88% to 92%.

2\. Mean Recall@5 increases from 72.67% to 76.67%.

3\. Mean Precision@10 remains unchanged at 68%.

4\. Recall@10 is 100% for both methods on the current manually labelled queries.

5\. MAP@10 improves from 87.37% to 95.11%.



These results indicate that attribute-aware ranking improves the ordering of relevant recommendations in the current evaluation set.



\*\*Evaluation limitation:\*\* The experiment uses five manually labelled queries. The relevance labels are based on catalogue metadata and relevance judgments, not an independent large-scale human evaluation of every product image. Therefore, the results are preliminary and should not be interpreted as proof of general superiority across all fashion categories.



\## 10. Streamlit Application Features



The application provides the following features:



\- Image-only product search.

\- Text-only product search.

\- Combined image and text search.

\- Category filtering.

\- Adjustable Top-K recommendation count.

\- Baseline and experimental ranking selection.

\- Product image display.

\- CLIP similarity scores.

\- Metadata matching scores.

\- Attribute matching scores.

\- Combined ranking scores.

\- CLIP similarity progress visualization.

\- Explanations describing why a product was recommended.

\- Input validation and error handling.



Similarity and ranking scores are retrieval signals. They are not calibrated probabilities or confidence values.



\## 11. Installation and Execution



\### Prerequisites



\- Python 3.10 or 3.11 recommended.

\- Internet connection for the first download of the pretrained CLIP model.

\- Sufficient memory and storage for the catalogue and model files.



\### Step 1: Clone or extract the project



Open PowerShell in the project directory.



\### Step 2: Create a virtual environment



```powershell

python -m venv .venv

```



\### Step 3: Activate the environment



```powershell

.venv\\Scripts\\Activate.ps1

```



\### Step 4: Install dependencies



```powershell

python -m pip install --upgrade pip

pip install -r requirements.txt

```



\### Step 5: Prepare the dataset (first-time setup only)



If you are starting with the original dataset files, prepare the catalogue using:



```powershell

python scripts\\prepare\_dataset.py

```



Use the dataset files and options expected by the script. If the prepared catalogue is already available, do not repeat this step.



\### Step 6: Build the FAISS index (first-time setup or after catalogue changes)



```powershell

python scripts\\build\_index.py

```



\### Step 7: Run the application



```powershell

streamlit run app.py

```



Streamlit will display a local URL. Open that URL in your browser to use FashionMatch AI.



\## 12. Running the Evaluation



Run the baseline evaluation:



```powershell

python scripts\\evaluate\_recommendations.py

```



Run the experimental evaluation:



```powershell

python scripts\\evaluate\_experimental.py

```



Generate the final comparison:



```powershell

python scripts\\create\_final\_comparison.py

```



The evaluation outputs are saved in the `data/` directory.



\## 13. Limitations



\- The current evaluation uses only five manually labelled search queries.

\- The relevance labels are not based on a large independent human evaluation.

\- Catalogue metadata quality can affect attribute matching.

\- CLIP similarity may not always distinguish fine-grained clothing attributes.

\- The current system does not implement a dedicated fashion object detector or clothing segmentation model.

\- The system does not perform purchases or connect to live retailer inventory.

\- Product prices are unavailable in the current catalogue.



\## 14. Future Enhancements



\- Expand the evaluation dataset with more queries and independently reviewed relevance labels.

\- Add human evaluation of recommendation quality.

\- Improve fine-grained attribute extraction.

\- Integrate clothing detection and segmentation.

\- Explore advanced multimodal reranking models.

\- Improve retrieval efficiency for larger catalogues.

\- Add user feedback and personalized recommendations.

\- Evaluate latency and scalability with larger product collections.



\## 15. Conclusion



FashionMatch AI demonstrates a multimodal fashion product retrieval and recommendation system using CLIP and FAISS.



The application supports image-based, text-based, and combined image-text searches. Two ranking methods were implemented and compared: a baseline CLIP-plus-metadata method and an experimental attribute-aware method.



On the current five-query evaluation set, the experimental method achieved 92% Mean Precision@5 and 95.11% MAP@10, compared with 88% Mean Precision@5 and 87.37% MAP@10 for the baseline.



The results provide preliminary evidence that incorporating product attributes can improve the ordering of fashion recommendations. Further evaluation using a larger and independently reviewed dataset is required to establish generalization.



\---



\*\*Project:\*\* FashionMatch AI  

\*\*Domain:\*\* Artificial Intelligence and Machine Learning  

\*\*Application:\*\* Multimodal Fashion Product Search and Recommendation  

\*\*Interface:\*\* Streamlit

