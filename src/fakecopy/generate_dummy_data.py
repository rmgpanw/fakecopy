import pandas as pd
import numpy as np

def generate_dummy_data_from_summary(summary, n_rows=100):
    dummy_data = {}

    for _, row in summary.iterrows():
        column = row['column']
        dtype = row['dtype']
        na_proportion = row['na_proportion']
        
        if 'float' in dtype:
            # Float columns
            mean, std = row['mean'], row['std']
            dummy_data[column] = np.random.normal(loc=mean, scale=std, size=n_rows)
        
        elif 'int' in dtype:
            # Integer columns
            mean, std = row['mean'], row['std']
            min_val, max_val = row['min'], row['max']
            dummy_data[column] = np.clip(np.random.normal(loc=mean, scale=std, size=n_rows).round(), min_val, max_val).astype(int)
        
        elif 'object' in dtype or 'category' in dtype:
            # Categorical or object columns
            categories = list(row['frequencies'].keys())
            probabilities = list(row['frequencies'].values())
            dummy_data[column] = np.random.choice(categories, size=n_rows, p=probabilities)
        
        elif 'bool' in dtype:
            # Boolean columns
            probabilities = row['frequencies']
            dummy_data[column] = np.random.choice([True, False], size=n_rows, p=[probabilities.get(True, 0.5), probabilities.get(False, 0.5)])
        
        elif 'datetime' in dtype:
            # Date or datetime columns
            min_date, max_date = row['min'], row['max']
            dummy_data[column] = pd.to_datetime(
                np.random.randint(min_date.value, max_date.value, size=n_rows)
            )
        
        else:
            # Fallback for any other types
            unique_vals = row.get('unique_values', [0, 1])
            dummy_data[column] = np.random.choice(unique_vals, size=n_rows)

        # Apply missing values based on NA proportion
        if na_proportion > 0:
            na_indices = np.random.choice(n_rows, int(n_rows * na_proportion), replace=False)
            dummy_data[column][na_indices] = np.nan

    # Create DataFrame
    dummy_df = pd.DataFrame(dummy_data)
    return dummy_df

# Example usage with an existing DataFrame `df`
from summarise_dataframe import summarize_dataframe

df = pd.DataFrame({
    'x': np.random.normal(10, 2, 50),
    'y': np.random.choice(['A', 'B', 'C'], 50),
    'z': np.random.randint(1, 10, 50),
    'is_active': np.random.choice([True, False], 50),
    'date': pd.date_range('2020-01-01', periods=50, freq='D')
})
# Introduce some missing values for testing
df.loc[df.sample(frac=0.2).index, 'x'] = np.nan
df.loc[df.sample(frac=0.1).index, 'y'] = np.nan

# Step 1: Summarize the DataFrame
summary_df = summarize_dataframe(df)
print(summary_df)

# Step 2: Generate dummy data from the summary
dummy_df = generate_dummy_data_from_summary(summary_df, n_rows=100)
print(dummy_df.head())
