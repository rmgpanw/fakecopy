import pytest
import pandas as pd
from fakecopy.summarise_dataframe import summarize_dataframe
from fakecopy.generate_dummy_data import generate_dummy_data_from_summary
from fakecopy.dummy_data import create_dummy_data

class TestGenerateDummyDataFromSummary:
    
    @pytest.fixture(scope="class")
    def df(self):
        return create_dummy_data(seed=42)
    
    def test_generate_dummy_data_from_summary(self, df):
        summary_df = summarize_dataframe(df)
        dummy_df = generate_dummy_data_from_summary(summary_df, n_rows=100)
        
        assert isinstance(dummy_df, pd.DataFrame)
        assert len(dummy_df) == 100
        assert set(dummy_df.columns) == set(df.columns)

        for column in df.columns:
            assert dummy_df[column].isna().mean() <= summary_df.loc[summary_df['column'] == column, 'na_proportion'].values[0]
