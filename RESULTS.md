# Results Summary

## Dataset

Dataset: Databricks OfficeQA Treasury Bulletin dataset  
Years used: 2015–2025  
Questions evaluated for retrieval: 17  

## Technical Stack

- Python
- ChromaDB
- SentenceTransformers embedding model: `all-MiniLM-L6-v2`
- Local LLM for generator testing: Ollama `llama3.2:3b`

## Retriever Metrics

| Metric | Baseline | Engineered |
|---|---:|---:|
| Hit Rate@5 | 17.65% | 52.94% |
| MRR | 0.054 | 0.485 |

## Generator Test

The generator was tested on 3 questions using a free local Ollama model.

| Metric | Value |
|---|---:|
| Factual Accuracy | 0.00% |
| Source-based answers | 2 |
| Grounded refusals | 1 |

## Engineering Reflection

### Bottleneck

The main baseline bottleneck was retrieval. The baseline retriever often found semantically similar Treasury tables from the wrong year or wrong section. This was shown by the low baseline Hit Rate@5 of 17.65% and MRR of 0.054.

### Metadata Fix

Adding Year and Month metadata improved retrieval. The engineered retrieval system improved Hit Rate@5 from 17.65% to 52.94% and MRR from 0.054 to 0.485. Metadata helped narrow the search space so the retriever was less likely to return similar tables from the wrong Treasury Bulletin.

### Scaling Insight

If this system were scaled from the 2015–2025 subset to the full 1939–2025 archive, the first bottleneck would likely be retrieval quality and vector database size. The number of chunks would increase significantly, making embedding, indexing, and querying slower. The pipeline would need stronger metadata filtering, hybrid search, table-aware chunking, and possibly reranking.

### Generator Limitation

The local Ollama model was useful for a free test, but it performed poorly on financial QA. It selected wrong values, misunderstood table units, and struggled with derived calculations. A stronger model or Python-based numeric calculation layer would be needed for reliable answer generation.