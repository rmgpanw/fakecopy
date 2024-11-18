import pytest
import pandas as pd
from fakecopy.dummy import DummyDF, _create_test_dummy_data

class TestSummarizeDataFrame:
    
    @pytest.fixture(scope="class")
    def df(self):
        return _create_test_dummy_data(seed=42)
    
    def test_summarize_dataframe(self, df):
        df_utils = DummyDF(data=df)
        summary_df = df_utils.summary
        
        assert isinstance(summary_df, pd.DataFrame)
        assert len(summary_df) == len(df.columns)
        assert 'column' in summary_df.columns
        assert 'dtype' in summary_df.columns
        assert 'na_proportion' in summary_df.columns

        for _, row in summary_df.iterrows():
            assert row['column'] in df.columns
            assert row['dtype'] is not None
            assert 0 <= row['na_proportion'] <= 1

class TestGenerateDummyDataFromSummary:
    
    @pytest.fixture(scope="class")
    def df(self):
        return _create_test_dummy_data(seed=42)
    
    def test_generate_dummy_data_from_summary(self, df):
        df_utils = DummyDF(data=df)
        summary_df = df_utils.summary
        dummy_df = df_utils.generate_dummy_data(n_rows=100)
        
        assert isinstance(dummy_df, pd.DataFrame)
        assert len(dummy_df) == 100
        assert set(dummy_df.columns) == set(df.columns)

        for column in df.columns:
            assert dummy_df[column].isna().mean() <= summary_df.loc[summary_df['column'] == column, 'na_proportion'].values[0]