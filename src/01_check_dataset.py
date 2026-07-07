from datasets import load_dataset

print("Loading OfficeQA answer key...")

dataset = load_dataset(
    "databricks/officeqa",
    data_files="officeqa_full.csv",
    split="train"
)

df = dataset.to_pandas()

print("Loaded successfully")
print("Shape:", df.shape)
print("Columns:")
print(df.columns.tolist())
print(df.head())