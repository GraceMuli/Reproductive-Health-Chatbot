import pandas as pd

# Function to load and count intents from a CSV file
def count_intents(csv_file):
    # Load the CSV file into a DataFrame
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found. Please check the file path.")
        return
    
    # Count occurrences of each intent
    intent_counts = df['intent'].value_counts().to_dict()
    
    return intent_counts

# Example usage
if __name__ == "__main__":
    csv_file = 'processed_dataset.csv'  # Adjust the file path as per your dataset
    
    # Count intents
    intent_counts = count_intents(csv_file)
    
    # Print intents and their counts
    if intent_counts:
        print("Intent Counts:")
        for intent, count in intent_counts.items():
            print(f"{intent}: {count}")
