import os
import pandas as pd
from data_cleaning import clean_dataset

# Path to your original dataset
file_path = '/Users/mattbaglietto/data_cleaning_utility/gold_prices.csv'
df = pd.read_csv(file_path)

# Clean the dataset with custom options, including max_unique_for_encoding
cleaned_df = clean_dataset(df, 
                           missing_threshold=0.3, 
                           scaling_method='minmax', 
                           outlier_method='IQR',
                           max_unique_for_encoding=10)  # Adjust as needed

# Display cleaned data summary
print("Cleaned Data Summary:")
print(cleaned_df.info())
print(cleaned_df.head())

# Generate output path with "cleaned_" prefix
base_name = os.path.basename(file_path)
name, ext = os.path.splitext(base_name)
output_path = f'/Users/mattbaglietto/data_cleaning_utility/cleaned_{name}{ext}'

# Save the cleaned dataset
cleaned_df.to_csv(output_path, index=False)
print(f"Cleaned dataset saved to: {output_path}")
