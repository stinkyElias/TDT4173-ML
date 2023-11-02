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

        # Create aggregated instances of the training data

        self.data_A_aggregated_values = ReadData('A')
        self.data_B_aggregated_values = ReadData('B')
        self.data_C_aggregated_values = ReadData('C')

        self.train_observed_A_aggregated_values = self.data_A_aggregated_values.import_train_observed_data()
        self.train_observed_B_aggregated_values = self.data_B_aggregated_values.import_train_observed_data()
        self.train_observed_C_aggregated_values = self.data_C_aggregated_values.import_train_observed_data()

        self.train_estimated_A_aggregated_values = self.data_A_aggregated_values.import_train_estimated_data()
        self.train_estimated_B_aggregated_values = self.data_B_aggregated_values.import_train_estimated_data()
        self.train_estimated_C_aggregated_values = self.data_C_aggregated_values.import_train_estimated_data()

        self.A_aggregated_values = ConcatinateTrainingData(self.train_observed_A_aggregated_values,
                                              self.train_estimated_A_aggregated_values)
        self.B_aggregated_values = ConcatinateTrainingData(self.train_observed_B_aggregated_values,
                                              self.train_estimated_B_aggregated_values)
        self.C_aggregated_values = ConcatinateTrainingData(self.train_observed_C_aggregated_values,
                                              self.train_estimated_C_aggregated_values)

        self.training_A_aggregated_values = self.A_aggregated_values.concatinate_training_data()
        self.training_B_aggregated_values = self.B_aggregated_values.concatinate_training_data()
        self.training_C_aggregated_values = self.C_aggregated_values.concatinate_training_data()
    
    def add_target_to_training_data(self, use_aggregated_values: bool) -> None:
        """
        Add the target data to the training data.

        Argument
        --------
        - use_aggregated_values : bool
            If True, the mean, median or sum of values of each hour are used instead of the values
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

        if use_aggregated_values == True:    
            self.training_A_aggregated_values['pv_measurement'] = target_A['pv_measurement']
            self.training_B_aggregated_values['pv_measurement'] = target_B['pv_measurement']
            self.training_C_aggregated_values['pv_measurement'] = target_C['pv_measurement']
        else:
            self.training_A['pv_measurement'] = target_A['pv_measurement']
            self.training_B['pv_measurement'] = target_B['pv_measurement']
            self.training_C['pv_measurement'] = target_C['pv_measurement']

    def create_training_data(self, use_mean_values: bool = False,
                             use_median_values: bool = False,
                             use_summed_values: bool = False) -> pd.DataFrame:
        """
        Returns a DataFrame with the training data from all locations. If mean
        is set to True, the mean values of each hour are used instead of the
        values from each 15 minutes.

        Argument
        --------
        - use_mean_values : bool
            If True, the mean values of each hour are used instead of the values
            from each 15 minutes.
        
        - use_median_values : bool
            If True, the median values of each hour are used instead of the values
            from each 15 minutes.
        
        - use_summed_values : bool
            If True, the sum of values of each hour are used instead of the values
            from each 15 minutes.
        """
        if use_mean_values == True or use_median_values == True or use_summed_values == True:
            if use_mean_values == True:
                self.training_A_aggregated_values = self.training_A_aggregated_values.resample('H').mean()
                self.training_B_aggregated_values = self.training_B_aggregated_values.resample('H').mean()
                self.training_C_aggregated_values = self.training_C_aggregated_values.resample('H').mean()
            
            elif use_median_values == True:
                self.training_A_aggregated_values = self.training_A_aggregated_values.resample('H').median()
                self.training_B_aggregated_values = self.training_B_aggregated_values.resample('H').median()
                self.training_C_aggregated_values = self.training_C_aggregated_values.resample('H').median()
            
            elif use_summed_values == True:
                self.training_A_aggregated_values = self.training_A_aggregated_values.resample('H').sum()
                self.training_B_aggregated_values = self.training_B_aggregated_values.resample('H').sum()
                self.training_C_aggregated_values = self.training_C_aggregated_values.resample('H').sum()

            self.add_target_to_training_data(use_aggregated_values=True)

            return pd.concat([self.training_A_aggregated_values, self.training_B_aggregated_values,
                                self.training_C_aggregated_values], axis=0)

        else:
            self.add_target_to_training_data(use_aggregated_values=False)

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

def create_test_data(use_mean_values: bool = False,
                     use_median_values: bool = False,
                     use_summed_values: bool = False) -> pd.DataFrame:
    """
    Returns a DataFrame with the test data from all locations.
    Index is set to 'date_forecast'.

    Argument
    --------
    - use_mean_values : bool
        If True, the mean values of each hour are used instead of the values
        from each 15 minutes.
    
    - use_median_values : bool
        If True, the median values of each hour are used instead of the values
        from each 15 minutes.
    
    - use_summed_values : bool
        If True, the sum of values of each hour are used instead of the values
        from each 15 minutes.
    """
    data_A = ReadData('A')
    data_B = ReadData('B')
    data_C = ReadData('C')

    test_A = data_A.import_test_estimated_data()
    test_B = data_B.import_test_estimated_data()
    test_C = data_C.import_test_estimated_data()

    if use_mean_values == True or use_median_values == True or use_summed_values == True:
        test_A.set_index('date_forecast', inplace=True)
        test_B.set_index('date_forecast', inplace=True)
        test_C.set_index('date_forecast', inplace=True)
        
        if use_mean_values == True:      
            test_A = test_A.resample('H').mean(numeric_only=False)
            test_B = test_B.resample('H').mean(numeric_only=False)
            test_C = test_C.resample('H').mean(numeric_only=False)

            test_data = pd.concat([test_A, test_B, test_C], axis=0)

            test_data.dropna(how='all', axis=0, inplace=True)
        
        elif use_median_values == True:          
            test_A = test_A.resample('H').median(numeric_only=False)
            test_B = test_B.resample('H').median(numeric_only=False)
            test_C = test_C.resample('H').median(numeric_only=False)

            test_data = pd.concat([test_A, test_B, test_C], axis=0)

            test_data.dropna(how='all', axis=0, inplace=True)
        
        elif use_summed_values == True:
            test_A = test_A.resample('H').sum(numeric_only=True)
            test_B = test_B.resample('H').sum(numeric_only=True)
            test_C = test_C.resample('H').sum(numeric_only=True)

            test_data = pd.concat([test_A, test_B, test_C], axis=0)

            test_data.dropna(how='all', axis=0, inplace=True)

            all_zeros = (test_data == 0).all(axis=1)
            test_data = test_data[~all_zeros]
           
    else:
        test_data = pd.concat([test_A, test_B, test_C], axis=0)
        test_data.set_index('date_forecast', inplace=True)

    return test_data