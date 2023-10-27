import pandas as pd

# from read_data import ReadData
from utilities.read_data import ReadData
from utilities.concatinate_training_data import ConcatinateTrainingData

class CreateTrainingData:
    """
    Concatinates the training data from all locations and returns it as a DataFrame.
    """
    def __init__(self):
        """
        Initialize a CreateTrainingData instance while using the ReadData and
        ConcatinateTrainingData classes.
        """
        self.data_A = ReadData('A')
        self.data_B = ReadData('B')
        self.data_C = ReadData('C')

        self.train_observed_A = self.data_A.import_train_observed_data()
        self.train_observed_B = self.data_B.import_train_observed_data()
        self.train_observed_C = self.data_C.import_train_observed_data()

        self.train_estimated_A = self.data_A.import_train_estimated_data()
        self.train_estimated_B = self.data_B.import_train_estimated_data()
        self.train_estimated_C = self.data_C.import_train_estimated_data()

        self.A = ConcatinateTrainingData(self.train_observed_A, self.train_estimated_A)
        self.B = ConcatinateTrainingData(self.train_observed_B, self.train_estimated_B)
        self.C = ConcatinateTrainingData(self.train_observed_C, self.train_estimated_C)

        self.training_A = self.A.concatinate_training_data()
        self.training_B = self.B.concatinate_training_data()
        self.training_C = self.C.concatinate_training_data()

        # Create mean instances of the training data

        self.data_A_mean = ReadData('A')
        self.data_B_mean = ReadData('B')
        self.data_C_mean = ReadData('C')

        self.train_observed_A_mean = self.data_A_mean.import_train_observed_data()
        self.train_observed_B_mean = self.data_B_mean.import_train_observed_data()
        self.train_observed_C_mean = self.data_C_mean.import_train_observed_data()

        self.train_estimated_A_mean = self.data_A_mean.import_train_estimated_data()
        self.train_estimated_B_mean = self.data_B_mean.import_train_estimated_data()
        self.train_estimated_C_mean = self.data_C_mean.import_train_estimated_data()

        self.A_mean = ConcatinateTrainingData(self.train_observed_A_mean,
                                              self.train_estimated_A_mean)
        self.B_mean = ConcatinateTrainingData(self.train_observed_B_mean,
                                              self.train_estimated_B_mean)
        self.C_mean = ConcatinateTrainingData(self.train_observed_C_mean,
                                              self.train_estimated_C_mean)

        self.training_A_mean = self.A_mean.concatinate_training_data()
        self.training_B_mean = self.B_mean.concatinate_training_data()
        self.training_C_mean = self.C_mean.concatinate_training_data()
    
    def add_target_to_training_data(self, use_mean_values: bool) -> None:
        """
        Add the target data to the training data.

        Argument
        --------
        - use_mean_values : bool
            If True, the mean values of each hour are used instead of the values
            from each 15 minutes.
        """
        A = ReadData('A')
        B = ReadData('B')
        C = ReadData('C')

        target_A = A.import_target_data()
        target_B = B.import_target_data()
        target_C = C.import_target_data()

        target_A.set_index('time', inplace=True)
        target_B.set_index('time', inplace=True)
        target_C.set_index('time', inplace=True)

        if use_mean_values == True:    
            self.training_A_mean['pv_measurement'] = target_A['pv_measurement']
            self.training_B_mean['pv_measurement'] = target_B['pv_measurement']
            self.training_C_mean['pv_measurement'] = target_C['pv_measurement']
        else:
            self.training_A['pv_measurement'] = target_A['pv_measurement']
            self.training_B['pv_measurement'] = target_B['pv_measurement']
            self.training_C['pv_measurement'] = target_C['pv_measurement']

    def create_training_data(self, use_mean_values: bool) -> pd.DataFrame:
        """
        Returns a DataFrame with the training data from all locations. If mean
        is set to True, the mean values of each hour are used instead of the
        values from each 15 minutes.

        Argument
        --------
        - use_mean_values : bool
            If True, the mean values of each hour are used instead of the values
            from each 15 minutes.
        """
        if use_mean_values == True:
            self.training_A_mean = self.training_A_mean.resample('H').mean()
            self.training_B_mean = self.training_B_mean.resample('H').mean()
            self.training_C_mean = self.training_C_mean.resample('H').mean()

            self.add_target_to_training_data(use_mean_values=use_mean_values)

            return pd.concat([self.training_A_mean, self.training_B_mean,
                              self.training_C_mean], axis=0)
        
        else:
            self.add_target_to_training_data(use_mean_values=use_mean_values)

            return pd.concat([self.training_A, self.training_B, self.training_C],
                             axis=0)

def create_target_data() -> pd.DataFrame:
    """
    Returns a DataFrame with the target data from all locations.
    Index is set to 'time'.
    """
    data_A = ReadData('A')
    data_B = ReadData('B')
    data_C = ReadData('C')

    target_A = data_A.import_target_data()
    target_B = data_B.import_target_data()
    target_C = data_C.import_target_data()

    target_data = pd.concat([target_A, target_B, target_C], axis=0)
    target_data.set_index('time', inplace=True)

    return target_data

def create_test_data(use_mean_values: bool) -> pd.DataFrame:
    """
    Returns a DataFrame with the test data from all locations.
    Index is set to 'date_forecast'.

    Argument
    --------
    - use_mean_values : bool
        If True, the mean values of each hour are used instead of the values
        from each 15 minutes.
    """
    data_A = ReadData('A')
    data_B = ReadData('B')
    data_C = ReadData('C')

    test_A = data_A.import_test_estimated_data()
    test_B = data_B.import_test_estimated_data()
    test_C = data_C.import_test_estimated_data()

    if use_mean_values == True:      
        test_A.set_index('date_forecast', inplace=True)
        test_B.set_index('date_forecast', inplace=True)
        test_C.set_index('date_forecast', inplace=True)
        
        test_A = test_A.resample('H').mean(numeric_only=False)
        test_B = test_B.resample('H').mean(numeric_only=False)
        test_C = test_C.resample('H').mean(numeric_only=False)

        test_data = pd.concat([test_A, test_B, test_C], axis=0)

        test_data.dropna(how='all', axis=0, inplace=True)
        # test_data.to_csv('test_data_mean_no_nans.csv')
        
    else:
        test_data = pd.concat([test_A, test_B, test_C], axis=0)
        test_data.set_index('date_forecast', inplace=True)

        # test_data.to_csv('test_data.csv')

    return test_data