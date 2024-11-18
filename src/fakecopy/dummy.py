import pandas as pd
import numpy as np

class DummyDF:
    """
    A utility class for summarizing a DataFrame and generating dummy data based on the summary.

    Attributes:
        summary (pd.DataFrame): A summary of the DataFrame.

    Methods:
        summarize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
            Summarizes the given DataFrame.
        
        generate_dummy_data(n_rows: int = 100) -> pd.DataFrame:
            Generates dummy data based on the summary.
        
        save_summary_to_csv(filepath: str) -> None:
            Saves the summary to a CSV file.

        create_dummy_data(seed: int = None) -> pd.DataFrame:
            Creates a dummy DataFrame with predefined columns and random data.

    Example usage:
        >>> df_data = pd.DataFrame({
        ...     'x': np.random.normal(10, 2, 50),
        ...     'y': np.random.choice(['A', 'B', 'C'], 50),
        ...     'z': np.random.randint(1, 10, 50),
        ...     'is_active': np.random.choice([True, False], 50),
        ...     'date': pd.date_range('2020-01-01', periods=50, freq='D')
        ... })
        >>> df_data.loc[df_data.sample(frac=0.2).index, 'x'] = np.nan
        >>> df_data.loc[df_data.sample(frac=0.1).index, 'y'] = np.nan

        # Initialize with DataFrame
        >>> df_dummy = DummyDF(df=df_data)
        >>> print(df_dummy.summary.head())
           column      dtype  unique_values       mean       std  min  max  \
        0       x    float64            NaN   9.796078  2.010634  5.0  15.0   
        1       y     object            NaN        NaN       NaN  NaN  NaN   
        2       z      int64            NaN   5.000000  2.581989  1.0   9.0   
        3 is_active       bool  [True, False]        NaN       NaN  NaN  NaN   
        4     date datetime64            NaN        NaN       NaN  NaN  NaN   

           frequencies  na_proportion  
        0          NaN            0.2  
        1          NaN            0.1  
        2          NaN            0.0  
        3          NaN            0.0  
        4          NaN            0.0  

        # Generate dummy data
        >>> df_synthetic = df_dummy.generate_dummy_data(n_rows=100)
        >>> print(df_synthetic.head())
                  x  y  z  is_active       date
        0  10.123456  A  5       True 2020-01-01
        1   9.876543  B  3      False 2020-01-02
        2  10.234567  C  7       True 2020-01-03
        3   9.765432  A  2      False 2020-01-04
        4  10.345678  B  6       True 2020-01-05

        # Save summary to CSV
        >>> df_dummy.save_summary_to_csv('summary.csv')

        # Create dummy data using static method
        >>> df_synthetic_static = DummyDF.create_dummy_data(seed=42)
        >>> print(df_synthetic_static.head())
           numeric categorical  integer  boolean       date
        0      9.0           A        1     True 2020-01-01
        1     11.0           B        2    False 2020-01-02
        2     10.0           C        3     True 2020-01-03
        3      8.0           A        4    False 2020-01-04
        4     12.0           B        5     True 2020-01-05
    """

    def __init__(self, data, read_csv_args=None):
        if isinstance(data, str):
            if read_csv_args is None:
                read_csv_args = {}
            self.df = pd.read_csv(data, **read_csv_args)
        elif isinstance(data, pd.DataFrame):
            if read_csv_args is not None:
                print("Warning: read_csv_args will be ignored as a DataFrame is provided.")
            self.df = data
        else:
            raise ValueError("data must be a file path (string) or a pandas DataFrame.")
        
        self.summary = self.summarize_dataframe(self.df)

    def summarize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
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
                col_summary['mean'] = df[column].mean()
                col_summary['std'] = df[column].std()
                col_summary['min'] = df[column].min()
                col_summary['max'] = df[column].max()
            
            elif pd.api.types.is_integer_dtype(dtype):
                col_summary['mean'] = df[column].mean()
                col_summary['std'] = df[column].std()
                col_summary['min'] = df[column].min()
                col_summary['max'] = df[column].max()
            
            elif isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(dtype):
                value_counts = df[column].value_counts(normalize=True)
                if len(value_counts) > 100:
                    value_counts = value_counts[:100]
                col_summary['unique_values'] = value_counts.index.tolist()
                col_summary['frequencies'] = value_counts.to_dict()
            
            elif pd.api.types.is_bool_dtype(dtype):
                col_summary['unique_values'] = [True, False]
                col_summary['frequencies'] = df[column].value_counts(normalize=True).to_dict()
            
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                col_summary['min'] = df[column].min()
                col_summary['max'] = df[column].max()

            summary.append(col_summary)
        
        summary_df = pd.DataFrame(summary)
        return summary_df

    def generate_dummy_data(self, n_rows: int = 100) -> pd.DataFrame:
        dummy_data = {}

        for _, row in self.summary.iterrows():
            column = row['column']
            dtype = row['dtype']
            na_proportion = row['na_proportion']
            
            if 'float' in dtype:
                mean, std = row['mean'], row['std']
                dummy_data[column] = np.random.normal(loc=mean, scale=std, size=n_rows)
            
            elif 'int' in dtype:
                mean, std = row['mean'], row['std']
                min_val, max_val = row['min'], row['max']
                dummy_data[column] = np.clip(np.random.normal(loc=mean, scale=std, size=n_rows).round(), min_val, max_val).astype(int)
            
            elif 'object' in dtype or 'category' in dtype:
                categories = list(row['frequencies'].keys())
                if len(categories) > 100:
                    categories = categories[:100]
                probabilities = list(row['frequencies'].values())
                dummy_data[column] = np.random.choice(categories, size=n_rows, p=probabilities)
            
            elif 'bool' in dtype:
                probabilities = row['frequencies']
                dummy_data[column] = np.random.choice([True, False], size=n_rows, p=[probabilities.get(True, 0.5), probabilities.get(False, 0.5)])
            
            elif 'datetime' in dtype:
                min_date, max_date = row['min'], row['max']
                dummy_data[column] = pd.to_datetime(
                    np.random.randint(min_date.value, max_date.value, size=n_rows)
                )
            
            else:
                unique_vals = row.get('unique_values', [0, 1])
                dummy_data[column] = np.random.choice(unique_vals, size=n_rows)

            if na_proportion > 0:
                na_indices = np.random.choice(n_rows, int(n_rows * na_proportion), replace=False)
                dummy_data[column][na_indices] = np.nan

        dummy_df = pd.DataFrame(dummy_data)
        return dummy_df

    def save_summary_to_csv(self, filepath: str) -> None:
        self.summary.to_csv(filepath, index=False)

def _create_test_dummy_data(seed: int = None) -> pd.DataFrame:
    """
    Creates a dummy DataFrame with predefined columns and random data for testing purposes.

    Args:
        seed (int, optional): Seed for the random number generator.

    Returns:
        pd.DataFrame: A DataFrame with dummy data.
    """
    if seed is not None:
        np.random.seed(seed)
    
    df = pd.DataFrame({
        'numeric': np.random.normal(10, 2, 50),
        'categorical': np.random.choice(['A', 'B', 'C'], 50),
        'integer': np.random.randint(1, 10, 50),
        'boolean': np.random.choice([True, False], 50),
        'date': pd.date_range('2020-01-01', periods=50, freq='D')
    })
    # Introduce some missing values for testing
    df.loc[df.sample(frac=0.2, random_state=seed).index, 'numeric'] = np.nan
    df.loc[df.sample(frac=0.1, random_state=seed).index, 'categorical'] = np.nan
    return df