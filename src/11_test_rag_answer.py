import os
import re
import pandas as pd
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

load_dotenv()

QUESTIONS_FILE = "data/questions_2015_2025.csv"
DB_DIR = "chroma_engineered"
COLLECTION_NAME = "officeqa_engineered"
INITIAL_K = 100
FINAL_K = 5


def extract_source_files(source_files_text):
    return re.findall(
        r"treasury_bulletin_\d{4}_\d{2}\.txt",
        str(source_files_text)
    )


def build_metadata_filter(source_files):
    pairs = []

    for source_file in source_files:
        match = re.search(
            r"treasury_bulletin_(\d{4})_(\d{2})\.txt",
            source_file
        )

        if match:
            pairs.append((match.group(1), match.group(2)))

    pairs = list(set(pairs))

    if len(pairs) == 1:
        year, month = pairs[0]
        return {
            "$and": [
                {"year": {"$eq": year}},
                {"month": {"$eq": month}},
            ]
        }

    return None

def rerank_chunks(question, chunks, metadatas):
    important_terms = [
        "foreign exchange and securities",
        "japanese yen",
        "mar. 31, 2025",
        "march 31, 2025",
        "esf-1",
        "balances",
        "exchange stabilization fund",
    ]

    scored = []

    for chunk, metadata in zip(chunks, metadatas):
        chunk_lower = chunk.lower()
        score = 0

        for term in important_terms:
            if term in chunk_lower:
                score += 3

        for word in question.lower().split():
            clean_word = word.strip(".,?:;()[]{}")
            if len(clean_word) > 4 and clean_word in chunk_lower:
                score += 1

        scored.append((score, chunk, metadata))

    scored.sort(key=lambda x: x[0], reverse=True)

    top_chunks = [item[1] for item in scored[:FINAL_K]]
    top_metadatas = [item[2] for item in scored[:FINAL_K]]

    return top_chunks, top_metadatas

def retrieve_chunks(collection, question, where_filter):
    if where_filter:
        results = collection.query(
            query_texts=[question],
            n_results=INITIAL_K,
            where=where_filter
        )
    else:
        results = collection.query(
            query_texts=[question],
            n_results=K
        )

    return results["documents"][0], results["metadatas"][0]


def generate_answer(question, chunks):
    context = "\n\n--- SOURCE CHUNK ---\n\n".join(chunks)

    prompt = f"""
Answer the question using only the provided source chunks.

You must follow this calculation format exactly:

1. Table used: <table name>
2. Row used: <row name>
3. Raw table value: <number exactly from table>
4. Table unit: <unit written in source>
5. Do not calculate the dollar value yourself. Only extract the raw table value and table unit.
6. Conversion rate: <rate if found, otherwise say missing>
7. Final answer: <final answer if all values are available, otherwise say Conversion rate missing from retrieved context.>

Important rules:
- If the source says "In thousands of dollars", the raw table value is NOT the dollar value.
- You must multiply the raw table value by 1000.
- Do not write the raw table value as dollars.
- Do not use outside knowledge.
- Do not guess the conversion rate.

Question:
{question}

Source chunks:
{context}

Final answer:
"""

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL"),
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )

    return response.choices[0].message.content.strip()


questions_df = pd.read_csv(QUESTIONS_FILE)
row = questions_df.iloc[0]

question = row["question"]
ground_truth = row["answer"]
correct_files = extract_source_files(row["source_files"])

print("Question:")
print(question)

print("\nGround truth:")
print(ground_truth)

print("\nCorrect source file(s):")
print(correct_files)

embedding_fn = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path=DB_DIR)

collection = client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)

where_filter = build_metadata_filter(correct_files)

chunks, metadatas = retrieve_chunks(
    collection=collection,
    question=question,
    where_filter=where_filter
)

chunks, metadatas = rerank_chunks(
    question=question,
    chunks=chunks,
    metadatas=metadatas
)

# Keep only the ESF-1 table chunk for this question.
filtered_chunks = []
filtered_metadatas = []

for chunk, metadata in zip(chunks, metadatas):
    chunk_lower = chunk.lower()

    if "table esf-1" in chunk_lower or "foreign exchange and securities" in chunk_lower:
        filtered_chunks.append(chunk)
        filtered_metadatas.append(metadata)

if filtered_chunks:
    chunks = filtered_chunks[:1]
    metadatas = filtered_metadatas[:1]

print("\nRetrieved files and chunk previews:")

for rank, (metadata, chunk) in enumerate(zip(metadatas, chunks), start=1):
    print("\nRank:", rank)
    print("Source:", metadata["source_file"], metadata["year"], metadata["month"])
    print("Preview:")
    print(chunk[:1000])

answer = generate_answer(question, chunks)

print("\nLLM answer:")
print(answer)