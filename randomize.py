import pandas as pd

# Load your dataset
df = pd.read_csv('processed_dataset.csv')

# Shuffle the dataset
shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save shuffled dataset
shuffled_df.to_csv('processed_dataset.csv', index=False)

print("Dataset shuffled and saved successfully.")
