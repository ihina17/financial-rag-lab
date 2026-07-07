import re
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

QUESTIONS_FILE = "data/questions_2015_2025.csv"
DB_DIR = "chroma_baseline"
COLLECTION_NAME = "officeqa_baseline"

K = 5

def extract_year_month_from_source_file(source_files):
    match = re.search(r"treasury_bulletin_(\d{4})_(\d{2})\.txt", str(source_files))

    if not match:
        return None, None

    return match.group(1), match.group(2)

questions_df = pd.read_csv(QUESTIONS_FILE)

first_question = questions_df.iloc[0]

question = first_question["question"]
answer = first_question["answer"]
source_files = first_question["source_files"]

year, month = extract_year_month_from_source_file(source_files)

print("Question:")
print(question)

print("\nGround truth answer:")
print(answer)

print("\nCorrect source file(s):")
print(source_files)

print("\nMetadata filter:")
print("Year:", year)
print("Month:", month)

embedding_fn = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path=DB_DIR)

collection = client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)

results = collection.query(
    query_texts=[question],
    n_results=K,
    where={
        "$and": [
            {"year": {"$eq": year}},
            {"month": {"$eq": month}},
        ]
    }
)

print("\nTop retrieved chunks after metadata filtering:")

for rank, metadata in enumerate(results["metadatas"][0], start=1):
    print("\nRank:", rank)
    print("Source file:", metadata["source_file"])
    print("Year:", metadata["year"])
    print("Month:", metadata["month"])

    chunk_text = results["documents"][0][rank - 1]
    print("Chunk preview:")
    print(chunk_text[:500])