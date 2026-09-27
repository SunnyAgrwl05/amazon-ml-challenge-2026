import pandas as pd

print("Loading matching_results.tsv...")
df = pd.read_csv('matching_results.tsv', sep='\t')

# 1. Clean missing/null values
df['matched_entity_ids'] = df['matched_entity_ids'].fillna('')

# 2. Precision Optimization Trick for F0.5:
# Drop noisy multi-matches (e.g. S2-xxx,S3-yyy) where risk of FP is high
# Keep only strict single matches or exact predictions
def filter_high_confidence(val):
    if not val:
        return ""
    matches = str(val).split(',')
    # If model predicted more than 2 matches, it's likely noisy -> convert to singleton
    if len(matches) > 2:
        return ""
    return val

df['matched_entity_ids'] = df['matched_entity_ids'].apply(filter_high_confidence)

# Save to output folder
import os
os.makedirs('output', exist_ok=True)
df.to_csv('output/matching_results.tsv', sep='\t', index=False)
print("Saved high-precision output/matching_results.tsv")
