from pathlib import Path
from datasets import load_dataset
from huggingface_hub import snapshot_download

YEARS = ["2022", "2023", "2024", "2025"]

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

print("Downloading OfficeQA full answer key...")

dataset = load_dataset(
    "databricks/officeqa",
    data_files="officeqa_full.csv",
    split="train"
)

df = dataset.to_pandas()
df.to_csv(DATA_DIR / "officeqa_full.csv", index=False)

print("Saved data/officeqa_full.csv")
print("Rows:", len(df))

print("Downloading Treasury Bulletin text files for 2022-2025...")

patterns = [
    f"treasury_bulletins_parsed/transformed/treasury_bulletin_{year}_*.txt"
    for year in YEARS
]

snapshot_download(
    repo_id="databricks/officeqa",
    repo_type="dataset",
    allow_patterns=patterns,
    local_dir=DATA_DIR / "officeqa_files",
)

print("Download complete.")