from h2o import H2OFrame
import pandas as pd

class CreateDelivery:
    def __init__(self, H2O_frame: H2OFrame, file_name: str):
        self.H2O_frame = H2O_frame
        self.file_name = file_name
        self.pandas_frame = None

    def create_delivery_file(self) -> None:
        """
        Create a delivery file and save it in results folder
        as a csv file.
        """
        self.pandas_frame = self.convert_H2O_to_pandas()
        self.set_negative_values_to_zeros()
        self.purge_unnecessary_rows()
        self.reformat_index()

        self.pandas_frame.to_csv(f"results/{self.file_name}.csv", index=False)
    
    def convert_H2O_to_pandas(self) -> pd.DataFrame:
        """
        Cast H2OFrame to Pandas DataFrame.
        """
        return self.H2O_frame.as_data_frame()
    
    def set_negative_values_to_zeros(self) -> None:
        """
        Set all negative values to zero.
        """
        self.pandas_frame[self.pandas_frame < 0] = 0
    
    def purge_unnecessary_rows(self) -> None:
        """
        Delete unnecessary rows so fit the delivery file format.
        After the header, keep the first row and then delete the next
        three rows.
        After purging, the DataFrame will be 3/4 of its original size.
        """
        self.pandas_frame = self.pandas_frame.iloc[::4, :]

    def reformat_index(self) -> None:
        self.pandas_frame.reset_index(drop=True, inplace=True)

