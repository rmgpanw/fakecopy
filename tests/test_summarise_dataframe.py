import pytest
import pandas as pd
from fakecopy.summarise_dataframe import summarize_dataframe
from fakecopy.dummy_data import create_dummy_data

class TestSummarizeDataFrame:
    
    @pytest.fixture(scope="class")
    def df(self):
        return create_dummy_data(seed=42)
    
    def test_summarize_dataframe(self, df):
        summary_df = summarize_dataframe(df)
        
        assert isinstance(summary_df, pd.DataFrame)
        assert len(summary_df) == len(df.columns)
        assert 'column' in summary_df.columns
        assert 'dtype' in summary_df.columns
        assert 'na_proportion' in summary_df.columns

        for _, row in summary_df.iterrows():
            assert row['column'] in df.columns
            assert row['dtype'] is not None
            assert 0 <= row['na_proportion'] <= 1
