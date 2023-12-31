import os
import pandas as pd

class ReadData:
    """
    Imports and read data from the specified location A, B or C.
    All functions returns the data as a DataFrame.
    """
    def __init__(self, location: str):
        """
        Initialize a ReadData instance for a specified location.

        Argument
        ----------
        - location : str
            The location of the data to be imported. Must be either 'A', 'B' or 'C'.
        """
        self.location = location
        self.home_dir = os.path.expanduser('~')
        self.data_raw_dir = 'Documents/TDT4173-ML/data_raw'
        
        self.path_to_target = os.path.join(self.home_dir, self.data_raw_dir, self.location,
                                           'parquet/train_targets.parquet')
        self.path_to_train_observed = os.path.join(self.home_dir, self.data_raw_dir, self.location,
                                                   'parquet/X_train_observed.parquet')
        self.path_to_train_estimated = os.path.join(self.home_dir, self.data_raw_dir, self.location,
                                                    'parquet/X_train_estimated.parquet')
        self.path_to_test_estimated = os.path.join(self.home_dir, self.data_raw_dir, self.location,
                                                   'parquet/X_test_estimated.parquet')

    def import_target_data(self) -> pd.DataFrame:
        """
        Import target data and return it as a DataFrame
        """
        data_frame = pd.read_parquet(self.path_to_target, engine='pyarrow')
        data_frame.sort_index(inplace=True)

        return data_frame
    
    def import_train_observed_data(self) -> pd.DataFrame:
        """
        Import observed training data and return it as a DataFrame
        """
        data_frame = pd.read_parquet(self.path_to_train_observed, engine='pyarrow')
        data_frame.sort_index(inplace=True)

        return data_frame

    def import_train_estimated_data(self) -> pd.DataFrame:
        """
        Import estimated training data and return it as a DataFrame
        """
        data_frame = pd.read_parquet(self.path_to_train_estimated, engine='pyarrow')
        data_frame.sort_index(inplace=True)

        return data_frame
    
    def import_test_estimated_data(self) -> pd.DataFrame:
        """
        Import estimated test data and return it as a DataFrame
        """
        data_frame = pd.read_parquet(self.path_to_test_estimated, engine='pyarrow')
        data_frame.sort_index(inplace=True)

        return data_frame