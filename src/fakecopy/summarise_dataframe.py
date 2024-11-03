import pandas as pd
import numpy as np

def summarize_dataframe(df):
    summary = []
    
    for column in df.columns:
        dtype = df[column].dtype
        col_summary = {
            'column': column,
            'dtype': str(dtype),
            'unique_values': None,
            'mean': None,
            'std': None,
            'min': None,
            'max': None,
            'frequencies': None,
            'na_proportion': df[column].isna().mean()
        }
        
        if pd.api.types.is_float_dtype(dtype):
            # Float (continuous) columns
            col_summary['mean'] = df[column].mean()
            col_summary['std'] = df[column].std()
            col_summary['min'] = df[column].min()
            col_summary['max'] = df[column].max()
        
        elif pd.api.types.is_integer_dtype(dtype):
            # Integer columns
            col_summary['mean'] = df[column].mean()
            col_summary['std'] = df[column].std()
            col_summary['min'] = df[column].min()
            col_summary['max'] = df[column].max()
        
        elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(dtype):
            # Categorical or object columns
            unique_vals = df[column].dropna().unique()
            col_summary['unique_values'] = unique_vals
            # Store frequencies for each category
            col_summary['frequencies'] = df[column].value_counts(normalize=True).to_dict()
        
        elif pd.api.types.is_bool_dtype(dtype):
            # Boolean (logical) columns
            col_summary['unique_values'] = [True, False]
            # Calculate frequencies for True and False
            col_summary['frequencies'] = df[column].value_counts(normalize=True).to_dict()
        
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            # Date or datetime columns
            col_summary['min'] = df[column].min()
            col_summary['max'] = df[column].max()

        summary.append(col_summary)
    
    # Convert to DataFrame for easier saving to CSV if needed
    summary_df = pd.DataFrame(summary)
    return summary_df

# Example usage with an existing DataFrame `df`
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
