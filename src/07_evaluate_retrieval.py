import re
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from tqdm import tqdm

QUESTIONS_FILE = "data/questions_2015_2025.csv"
DB_DIR = "chroma_baseline"
COLLECTION_NAME = "officeqa_baseline"
K = 5


def extract_source_files(source_files_text):
    return re.findall(
        r"treasury_bulletin_\d{4}_\d{2}\.txt",
        str(source_files_text)
    )


def extract_year_month_pairs(source_files):
    pairs = []

    for source_file in source_files:
        match = re.search(
            r"treasury_bulletin_(\d{4})_(\d{2})\.txt",
            source_file
        )

        if match:
            pairs.append((match.group(1), match.group(2)))

    return list(set(pairs))


def build_metadata_filter(source_files):
    pairs = extract_year_month_pairs(source_files)

    if len(pairs) == 1:
        year, month = pairs[0]
        return {
            "$and": [
                {"year": {"$eq": year}},
                {"month": {"$eq": month}},
            ]
        }

    # If question uses multiple source files/months, skip metadata filter for now
    return None


def calculate_hit_and_rr(retrieved_files, correct_files):
    correct_files = set(correct_files)

    for rank, retrieved_file in enumerate(retrieved_files, start=1):
        if retrieved_file in correct_files:
            return 1, 1 / rank

    return 0, 0


def run_retrieval(collection, question, where_filter=None):
    if where_filter:
        results = collection.query(
            query_texts=[question],
            n_results=K,
            where=where_filter
        )
    else:
        results = collection.query(
            query_texts=[question],
            n_results=K
        )

    retrieved_files = [
        metadata["source_file"]
        for metadata in results["metadatas"][0]
    ]

    return retrieved_files


questions_df = pd.read_csv(QUESTIONS_FILE)

embedding_fn = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path=DB_DIR)

collection = client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)

baseline_hits = []
baseline_rrs = []

metadata_hits = []
metadata_rrs = []

details = []

for _, row in tqdm(questions_df.iterrows(), total=len(questions_df)):
    question = row["question"]
    correct_files = extract_source_files(row["source_files"])

    baseline_files = run_retrieval(
        collection=collection,
        question=question,
        where_filter=None
    )

    baseline_hit, baseline_rr = calculate_hit_and_rr(
        baseline_files,
        correct_files
    )

    where_filter = build_metadata_filter(correct_files)

    metadata_files = run_retrieval(
        collection=collection,
        question=question,
        where_filter=where_filter
    )

    metadata_hit, metadata_rr = calculate_hit_and_rr(
        metadata_files,
        correct_files
    )

    baseline_hits.append(baseline_hit)
    baseline_rrs.append(baseline_rr)

    metadata_hits.append(metadata_hit)
    metadata_rrs.append(metadata_rr)

    details.append({
        "uid": row["uid"],
        "question": question,
        "correct_files": correct_files,
        "baseline_files": baseline_files,
        "baseline_hit": baseline_hit,
        "baseline_rr": baseline_rr,
        "metadata_files": metadata_files,
        "metadata_hit": metadata_hit,
        "metadata_rr": metadata_rr,
        "metadata_filter_used": where_filter is not None,
    })

baseline_hit_rate = sum(baseline_hits) / len(baseline_hits)
baseline_mrr = sum(baseline_rrs) / len(baseline_rrs)

metadata_hit_rate = sum(metadata_hits) / len(metadata_hits)
metadata_mrr = sum(metadata_rrs) / len(metadata_rrs)

print("\nRetrieval Metrics")
print("-----------------")
print(f"Total questions: {len(questions_df)}")

print("\nBaseline")
print(f"Hit Rate@5: {baseline_hit_rate:.2%}")
print(f"MRR:        {baseline_mrr:.3f}")

print("\nMetadata-filtered")
print(f"Hit Rate@5: {metadata_hit_rate:.2%}")
print(f"MRR:        {metadata_mrr:.3f}")

results_df = pd.DataFrame(details)
results_df.to_csv("results/retrieval_results.csv", index=False)

print("\nSaved: results/retrieval_results.csv")