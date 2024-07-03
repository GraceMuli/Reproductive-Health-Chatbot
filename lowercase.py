import pandas as pd

# Load the dataset with the first column containing all data
df = pd.read_csv('processed_dataset.csv', header=None)

# Lowercase all text in the first column
df[0] = df[0].str.lower()

# Save the DataFrame to a new CSV file
df.to_csv('processed_dataset.csv', index=False, header=False)
