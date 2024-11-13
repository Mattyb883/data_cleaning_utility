# Data Cleaning Utility

## Overview
This repository contains a reusable, plug-and-play data cleaning function, `clean_dataset`, built for preprocessing new datasets with minimal configuration. The function handles missing values, removes duplicates, standardizes column names, handles outliers, encodes categorical data, scales numerical data, and more.

The `clean_dataset` function is designed to simplify data cleaning tasks, making it easy to apply consistent cleaning steps across multiple datasets by updating only the input file path in `example_usage.py`.

## Features
- **Automatic Handling of Missing Values**: Drops columns with high missing rates and fills remaining missing data with appropriate values.
- **Outlier Detection and Removal**: Supports Interquartile Range (IQR) and Z-score methods.
- **Text Standardization**: Converts text to lowercase and removes extra whitespace.
- **ISO 8601 Date Formatting**: Automatically converts date columns to `YYYY-MM-DD` format, ensuring a standardized and universally readable date format.
- **Flexible Date Handling**: Checks for column existence before processing, ensuring compatibility across datasets without modification.
- **Numerical Scaling**: Scales numerical features using either Standardization or Min-Max scaling.
- **Categorical Encoding with Customizable Limits**: One-hot encodes categorical features, with a configurable `max_unique_for_encoding` parameter to limit encoding to columns with a manageable number of unique values.
- **Dynamic Saving of Cleaned Dataset**: Saves the cleaned dataset with a `cleaned_` prefix automatically.

## Usage

### Prerequisites
This script requires the following Python libraries:
- `pandas`
- `numpy`
- `scikit-learn`

Install these with the following command:

```bash
pip install pandas numpy scikit-learn
