import re
import pandas as pd
from pathlib import Path

YEARS = [str(year) for year in range(2015, 2026)]

DATA_DIR = Path("data")
input_file = DATA_DIR / "officeqa_full.csv"
output_file = DATA_DIR / "questions_2015_2025.csv"

df = pd.read_csv(input_file)

def extract_years(source_files):
    source_files = str(source_files)
    return set(re.findall(r"treasury_bulletin_(\d{4})_\d{2}\.txt", source_files))

df["source_years"] = df["source_files"].apply(extract_years)

# Strict filter: keep only questions where all required source years are inside 2015-2025
filtered = df[
    df["source_years"].apply(lambda years: len(years) > 0 and years.issubset(YEARS))
].copy()

print("Total questions:", len(df))
print("Filtered questions:", len(filtered))

print("\nYears found in filtered questions:")
print(filtered["source_years"].value_counts())

filtered.to_csv(output_file, index=False)

print(f"\nSaved: {output_file}")