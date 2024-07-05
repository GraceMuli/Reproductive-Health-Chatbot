import pandas as pd

# Step 1: Load the CSV file into a DataFrame
csv_file = 'reproductive_dataset.csv'  # Replace with your actual CSV file path
df = pd.read_csv(csv_file, header=None)  # Assuming no header in the CSV file

# Step 2: Split each row based on the first two commas
split_data = df[0].str.split(',', n=2, expand=True)  # n=2 limits splitting to first two commas

# Step 3: Assign new column names
split_data.columns = ['question', 'intent', 'answer']

# Step 4: Optionally, save the processed DataFrame to a new CSV file
processed_csv_file = 'processed_dataset.csv'
split_data.to_csv(processed_csv_file, index=False)

print("CSV file saved successfully as:", processed_csv_file)
