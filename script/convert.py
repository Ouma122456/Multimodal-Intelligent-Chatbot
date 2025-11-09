import pandas as pd

# Load your parquet file
df = pd.read_parquet("validation-00000-of-00001.parquet")

# Save it as JSONL (records = one JSON per line)
df.to_json("validation.jsonl", orient="records", lines=True)

print("✅ Conversion complete! Saved as output.jsonl")
