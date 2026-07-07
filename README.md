# Financial RAG Lab

This project builds a Retrieval-Augmented Generation (RAG) system using the Databricks OfficeQA Treasury Bulletin dataset. The goal is to compare a simple baseline RAG pipeline with an engineered RAG pipeline and measure how engineering choices affect retrieval and answer quality.

## Project Goal

The assignment asks for a RAG system that searches through U.S. Treasury records and answers financial questions. This project implements:

1. A **Baseline RAG system**
2. An **Engineered RAG system**
3. Retrieval evaluation using **Hit Rate@5** and **MRR**
4. Generator testing using a local free LLM through **Ollama**

The main comparison is between a simple retrieval pipeline and an improved retrieval pipeline using better chunking and metadata filtering.

---

## Dataset

Dataset: Databricks OfficeQA Treasury Bulletin dataset
Data format used: transformed `.txt` Treasury Bulletin files
Years used: **2015–2025**
Question file: `officeqa_full.csv`

The project filters the OfficeQA answer key to keep questions whose source documents are from 2015–2025.

The downloaded dataset files are stored locally in the `data/` folder and are not pushed to GitHub.

---

## Technical Stack

* Python
* Pandas
* ChromaDB
* SentenceTransformers
* Embedding model: `all-MiniLM-L6-v2`
* Local LLM testing: Ollama `llama3.2:3b`
* OpenAI-compatible API client for local Ollama calls

---

## Project Structure

```text
financial-rag-lab/
│
├── README.md
├── RESULTS.md
├── requirements.txt
├── .gitignore
│
└── src/
    ├── 01_check_dataset.py
    ├── 02_download_data.py
    ├── 03_filter_questions.py
    ├── 04_build_baseline_db.py
    ├── 05_test_baseline_retrieval.py
    ├── 06_test_baseline_retrieval.py
    ├── 07_evaluate_retrieval.py
    ├── 08_build_engineered_db.py
    ├── 09_evaluate_engineered_retrieval.py
    ├── 10_check_llm_api.py
    ├── 11_test_rag_answer.py
    ├── 12_generate_answers.py
    └── 13_score_generated_answers.py
```

---

## Files Not Included in GitHub

The following files and folders are intentionally ignored:

```text
.env
data/
results/
chroma_baseline/
chroma_engineered/
.venv/
__pycache__/
*.pyc
```

These are excluded because they contain local data, generated outputs, API keys, or large vector database files.

---

## Baseline RAG System

The baseline system uses a simple retrieval setup.

### Baseline Design

* Fixed-size character chunking
* Chunk size: 1200 characters
* Overlap: 150 characters
* ChromaDB vector database
* SentenceTransformer embeddings
* No metadata filtering during retrieval

### Baseline Purpose

The baseline is intentionally simple. It shows how the system performs before applying engineering improvements.

---

## Engineered RAG System

The engineered system improves the baseline by changing how documents are chunked and retrieved.

### Engineered Design

* Paragraph-aware chunking
* Chunk size: 800 characters
* Overlap: 200 characters
* Year and Month metadata for every chunk
* Metadata filtering when the question maps to a specific source month
* Basic reranking for local generator testing

### Metadata Used

Each chunk is stored with:

```text
source_file
year
month
chunk_index
```

This metadata helps narrow the search space. For example, if a question’s answer is in `treasury_bulletin_2025_06.txt`, the engineered retriever can filter chunks to year `2025` and month `06`.

---

## Retriever Metrics

The retriever was evaluated on 17 OfficeQA questions from 2015–2025 using `K=5`.

| Metric     | Baseline | Engineered |
| ---------- | -------: | ---------: |
| Hit Rate@5 |   17.65% |     52.94% |
| MRR        |    0.054 |      0.485 |

### Interpretation

The engineered system significantly improved retrieval. Hit Rate@5 increased from 17.65% to 52.94%, and MRR increased from 0.054 to 0.485.

This shows that the main baseline failure was in finding the correct document or table, not only in answer generation.

---

## Generator Test

The generator was tested using a free local LLM through Ollama.

### Local Model

```text
llama3.2:3b
```

### Generator Test Results

The generator was tested on 3 questions.

| Metric               | Value |
| -------------------- | ----: |
| Questions tested     |     3 |
| Factual Accuracy     | 0.00% |
| Source-based answers |     2 |
| Grounded refusals    |     1 |

### Generator Limitation

The local Ollama model was useful for testing the pipeline, but it was weak for financial question answering. It often selected incorrect values, misunderstood table units, or failed on derived calculations.

For example, some OfficeQA questions require multi-step numerical reasoning, unit conversion, or outside conversion rates. A small local model was not reliable enough for those tasks.

---

## How to Run the Project

### 1. Create Virtual Environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

---

### 2. Install Requirements

```powershell
pip install -r requirements.txt
```

---

### 3. Login to Hugging Face

The OfficeQA dataset is gated, so access must be accepted on Hugging Face first.

After access is approved, authenticate locally:

```powershell
huggingface-cli login
```

or, if using the newer CLI:

```powershell
hf auth login
```

---

### 4. Check Dataset Access

```powershell
python src/01_check_dataset.py
```

This checks that the OfficeQA answer key can be loaded.

---

### 5. Download Data

```powershell
python src/02_download_data.py
```

This downloads the OfficeQA answer key and Treasury Bulletin text files.

---

### 6. Filter Questions

```powershell
python src/03_filter_questions.py
```

This creates a filtered question file for 2015–2025.

---

### 7. Build Baseline Vector Database

```powershell
python src/04_build_baseline_db.py
```

This creates the baseline ChromaDB vector database.

---

### 8. Test Baseline Retrieval

```powershell
python src/05_test_baseline_retrieval.py
```

---

### 9. Test Metadata-Based Retrieval

```powershell
python src/06_test_baseline_retrieval.py
```

---

### 10. Evaluate Baseline and Metadata Retrieval

```powershell
python src/07_evaluate_retrieval.py
```

---

### 11. Build Engineered Vector Database

```powershell
python src/08_build_engineered_db.py
```

---

### 12. Evaluate Engineered Retrieval

```powershell
python src/09_evaluate_engineered_retrieval.py
```

---

## Optional: Local LLM Generator with Ollama

This project uses Ollama as a free local model option.

### Install Ollama

Download Ollama from:

```text
https://ollama.com/download/windows
```

Then pull the model:

```powershell
ollama pull llama3.2:3b
```

---

### Configure `.env`

Create a `.env` file in the project root:

```text
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=llama3.2:3b
```

The `.env` file is ignored by GitHub.

---

### Check Local LLM API

```powershell
python src/10_check_llm_api.py
```

Expected output:

```text
API working
```

---

### Test RAG Answer Generation

```powershell
python src/11_test_rag_answer.py
```

---

### Generate Answers

```powershell
python src/12_generate_answers.py
```

---

### Score Generated Answers

```powershell
python src/13_score_generated_answers.py
```

---

## Engineering Reflection

### Bottleneck

The main bottleneck in the baseline system was retrieval. The baseline retriever often returned semantically similar Treasury tables from the wrong year or wrong section. This caused low Hit Rate@5 and MRR.

The baseline Hit Rate@5 was only 17.65%, and the MRR was 0.054. These values show that the correct document was often missing from the top 5 retrieved chunks or ranked very low.

---

### Metadata Fix

Adding Year and Month metadata improved retrieval. The engineered retriever used metadata to narrow the search space before vector search. This helped the system avoid retrieving similar Treasury tables from incorrect years.

The engineered system improved Hit Rate@5 from 17.65% to 52.94% and MRR from 0.054 to 0.485.

This shows that metadata helped the retriever more than the generator. The generator still struggled even when retrieval improved.

---

### Scaling Insight

If this system were scaled from the 2015–2025 subset to the full 1939–2025 archive, the first bottleneck would likely be retrieval quality and vector database size.

The full archive would create many more chunks. This would make embedding, indexing, and retrieval slower. It would also increase the chance of retrieving similar tables from incorrect years.

To scale this system, the pipeline would need:

* stronger metadata filtering
* hybrid search
* better table-aware chunking
* reranking
* numeric computation outside the LLM
* stronger model support for financial QA

---

## Conclusion

The engineered retriever clearly improved search performance compared with the baseline system. However, the generator remained weak when using a small free local model.

The main lesson is that RAG performance depends on both retrieval and generation. Metadata filtering can significantly improve retrieval, but financial QA also needs reliable table extraction, unit handling, and numeric computation.
