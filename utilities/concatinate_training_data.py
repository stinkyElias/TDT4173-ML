import pandas as pd

class ConcatinateTrainingData:
    """
    Creates training data (X train) by concatinating the observed and
    estimated X_train data.
    """
    def __init__(self, train_observed: pd.DataFrame, train_estimated: pd.DataFrame):
        """
        Initialize a ConcatinateTrainingData instance with the observed and estimated
        training data.

        Arguments
        ----------
        - train_observed : pd.DataFrame
            The observed training data.
        
        - train_estimated : pd.DataFrame
            The estimated training data.
        """
        self.train_observed = train_observed
        self.train_estimated = train_estimated
    
    def delete_date_calc(self) -> None:
        """
        Delete the date_calc column from the estimated training data.
        """
        self.train_estimated.drop(columns=['date_calc'], inplace=True)
    
    def format_time_index(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Convert the date_forecast column to a datetime object and set it as the index.
        """
        data_frame.set_index('date_forecast', inplace=True)
        
        return data_frame
    
    def concatinate_training_data(self) -> pd.DataFrame:
        """
        Returns the training data as a DataFrame where observed and estimated
        have no 'date_calc' column, 'date_forecast' is the index and estimated
        and observed are concatenated.
        """
        self.delete_date_calc()
        df = pd.concat([self.train_observed, self.train_estimated],
                                            axis=0)
        concatinated_training_data = self.format_time_index(df)

        return concatinated_training_data