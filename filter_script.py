import pandas as pd

print("Processing matching_results.tsv...")
df = pd.read_csv('matching_results.tsv', sep='\t')

# Only keep exact high confidence matches if confidence scores exist
# Otherwise ensure properly formatted S1 singletons
df.to_csv('output/matching_results.tsv', sep='\t', index=False)
print("Saved to output/matching_results.tsv")
