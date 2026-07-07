from pathlib import Path
import re
import shutil
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from tqdm import tqdm

TEXT_DIR = Path("data/officeqa_files/treasury_bulletins_parsed/transformed")
DB_DIR = Path("chroma_engineered")

COLLECTION_NAME = "officeqa_engineered"


def extract_year_month(filename):
    match = re.search(r"treasury_bulletin_(\d{4})_(\d{2})\.txt", filename)

    if not match:
        return None, None

    return match.group(1), match.group(2)


def chunk_text(text, chunk_size=800, overlap=200):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= chunk_size:
            current_chunk += "\n\n" + paragraph
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            overlap_text = current_chunk[-overlap:] if current_chunk else ""
            current_chunk = overlap_text + "\n\n" + paragraph

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

if DB_DIR.exists():
    shutil.rmtree(DB_DIR)

embedding_fn = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path=str(DB_DIR))

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)

txt_files = list(TEXT_DIR.glob("*.txt"))

print("Text directory:", TEXT_DIR)
print("Files found:", len(txt_files))

ids = []
documents = []
metadatas = []

for file_path in txt_files:
    year, month = extract_year_month(file_path.name)

    text = file_path.read_text(encoding="utf-8", errors="ignore")
    chunks = chunk_text(text)

    print(file_path.name, "Year:", year, "Month:", month, "Chunks:", len(chunks))

    for chunk_index, chunk in enumerate(chunks):
        ids.append(f"{file_path.stem}_chunk_{chunk_index}")
        documents.append(chunk)
        metadatas.append({
            "source_file": file_path.name,
            "year": year,
            "month": month,
            "chunk_index": chunk_index,
        })

print("Total chunks:", len(documents))
print("Adding chunks to ChromaDB...")

batch_size = 500

for start in tqdm(range(0, len(documents), batch_size)):
    end = start + batch_size

    collection.add(
        ids=ids[start:end],
        documents=documents[start:end],
        metadatas=metadatas[start:end],
    )

print("Engineered database created.")
print("Collection count:", collection.count())

