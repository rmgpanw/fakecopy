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

    def test_str_representation(self, df):
        df_utils = DummyDF(data=df)
        str_repr = str(df_utils)
        assert str_repr.startswith("Summary:")
        assert "column" in str_repr

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

class TestDummyDFInitialization:
    
    @pytest.fixture(scope="class")
    def df(self):
        return _create_test_dummy_data(seed=42)
    
    @pytest.fixture(scope="class")
    def csv_file(self, tmp_path_factory):
        df = _create_test_dummy_data(seed=42)
        csv_file = tmp_path_factory.mktemp("data") / "dummy_data.csv"
        df.to_csv(csv_file, index=False)
        return csv_file

    def test_initialization_with_dataframe(self, df):
        df_utils = DummyDF(data=df)
        assert isinstance(df_utils.summary, pd.DataFrame)
        assert not df_utils.summary.empty

    def test_initialization_with_csv_file(self, csv_file):
        df_utils = DummyDF(data=str(csv_file))
        assert isinstance(df_utils.summary, pd.DataFrame)
        assert not df_utils.summary.empty

    def test_initialization_with_invalid_data(self):
        with pytest.raises(ValueError):
            DummyDF(data=12345)