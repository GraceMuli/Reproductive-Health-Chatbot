import pandas as pd
from sklearn.model_selection import train_test_split

# Load the processed dataset
df = pd.read_csv("processed_dataset.csv")

# Split the dataset into train and test sets (80% train, 20% test)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save the train and test sets to separate CSV files
train_df.to_csv("train_reproductive_dataset.csv", index=False)
test_df.to_csv("test_reproductive_dataset.csv", index=False)
