import pandas as pd
import numpy as np

def create_dummy_data(seed=None):
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
