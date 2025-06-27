import pandas as pd
import ast

csv_path = "outputs/player_mappings.csv"
df = pd.read_csv(csv_path)

print("🔎 Sample rows:")
print(df.head())

# Try parsing one row
try:
    bbox = ast.literal_eval(df["broadcast_bbox"][0])
    print("✅ Parsed broadcast_bbox:", bbox)
except Exception as e:
    print("❌ Failed to parse broadcast_bbox:", e)
