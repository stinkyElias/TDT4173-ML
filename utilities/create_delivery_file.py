import os
import pandas as pd

from h2o.frame import H2OFrame

class CreateDelivery:
    """
    A class for creating a delivery file in the correct format and saving it as a csv file.
    """
    def __init__(self, H2O_frame: H2OFrame, file_name: str):
        """
        Initializes the CreateDelivery class.

        Arguments
        ---------
        - H2O_frame : H2OFrame
            An H2OFrame object containing the prediction data to be converted to a delivery file
        
        - file_name : str
            A string representing the name of the file to be saved
        """
        self.H2O_frame = H2O_frame
        self.pandas_frame = None
        self.saved_file_path = os.path.join(os.path.expanduser('~'),
                                'Documents/TDT4173-ML/data_results', file_name)

    def create_delivery_file(self, use_mean_values: bool) -> None:
        """
        Creates a delivery file in the correct format and saves it in the results folder as a csv file.
        """
        self.pandas_frame = self.convert_H2O_to_pandas()
        self.set_negative_values_to_zeros()

        if use_mean_values == False:
            self.purge_unnecessary_rows()
        
        self.reformat_index()

        self.pandas_frame.insert(0, 'id', range(0, len(self.pandas_frame)))
        self.pandas_frame.rename(columns={'predict': 'predictions'}, inplace=True)
        self.pandas_frame.to_csv(os.path.join(self.saved_file_path), index=False)
    
    def convert_H2O_to_pandas(self) -> pd.DataFrame:
        """
        Casts H2OFrame to Pandas DataFrame.

        Returns:
        - A Pandas DataFrame object containing the data from the H2OFrame object.
        """
        return self.H2O_frame.as_data_frame()
    
    def set_negative_values_to_zeros(self) -> None:
        """
        Sets all negative values in the Pandas DataFrame to zero.
        """
        self.pandas_frame[self.pandas_frame < 0] = 0
    
    def purge_unnecessary_rows(self) -> None:
        """
        Deletes unnecessary rows to fit the delivery file format.
        After the header, keeps the first row and then deletes the next
        three rows.
        After purging, the DataFrame will be 3/4 of its original size.
        """
        self.pandas_frame = self.pandas_frame.iloc[::4, :]

    def reformat_index(self) -> None:
        """
        Resets the index of the Pandas DataFrame to start from 0.
        """
        self.pandas_frame.reset_index(drop=True, inplace=True)

