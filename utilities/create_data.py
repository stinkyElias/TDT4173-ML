import pandas as pd

from utilities.read_data import ReadData
from utilities.concatinate_training_data import ConcatinateTrainingData

class CreateData:
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

        self.test_A = self.data_A.import_test_estimated_data()
        self.test_B = self.data_B.import_test_estimated_data()
        self.test_C = self.data_C.import_test_estimated_data()
    
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
                             use_summed_values: bool = False,
                             impute_data: bool = False) -> pd.DataFrame:
        """
        Returns a DataFrame with the training data from all locations. All rows
        where all features are NaN are dropped, including all rows missing values
        in the target column.
        Feature 'snow_density:kgm3' is dropped because we miss 96% of the values.

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
        
        - impute_data : bool
            If True, missing values are imputed using KNN imputation.
        """
        if use_mean_values == True or use_median_values == True or use_summed_values == True:
            if impute_data == True:
                self.impute_data(data_A=self.training_A_aggregated_values, 
                                 data_B=self.training_B_aggregated_values,
                                 data_C=self.training_C_aggregated_values,
                                 use_basic_values=False)
            
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

            training_data = pd.concat([self.training_A_aggregated_values,
                                       self.training_B_aggregated_values,
                                       self.training_C_aggregated_values], axis=0)

        else:
            if impute_data == True:
                self.impute_data(data_A=self.training_A, data_B=self.training_B,
                                 data_C=self.training_C, use_basic_values=True)

            self.add_target_to_training_data(use_aggregated_values=False)

            training_data = pd.concat([self.training_A, self.training_B, self.training_C],
                             axis=0)
        
        feature_list = list(training_data.columns)
        feature_list.remove('pv_measurement')

        training_data.dropna(how='all', subset=feature_list, axis=0, inplace=True)
        training_data.dropna(how='all', subset=['pv_measurement'], axis=0, inplace=True)
    
        training_data.drop('snow_density:kgm3', axis=1, inplace=True)

        return training_data
    
    def impute_data(self, data_A: pd.DataFrame,
                    data_B: pd.DataFrame,
                    data_C: pd.DataFrame,
                    use_basic_values: bool = False,
                    for_training: bool = True) -> None:
        """
        Imputes missing values in the training data using KNN imputation.
        
        Argument
        --------
        - use_basic_values : bool
            If True, the training sets wihthout any resampling will be used.
            If False, the training sets with resampled values will be used.
        """
        from sklearn.impute import KNNImputer

        imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')

        data_A_missing = data_A[['ceiling_height_agl:m','cloud_base_agl:m']]
        data_B_missing = data_B[['ceiling_height_agl:m','cloud_base_agl:m']]
        data_C_missing = data_C[['ceiling_height_agl:m','cloud_base_agl:m']]

        data_A.drop(['ceiling_height_agl:m','cloud_base_agl:m'],
                                axis=1, inplace=True)
        data_B.drop(['ceiling_height_agl:m','cloud_base_agl:m'],
                                axis=1, inplace=True)
        data_C.drop(['ceiling_height_agl:m','cloud_base_agl:m'],
                                axis=1, inplace=True)

        imputed_data_A = data_A_missing.copy()
        imputed_data_B = data_B_missing.copy()
        imputed_data_C = data_C_missing.copy()

        imputed_data_A[['ceiling_height_agl:m','cloud_base_agl:m']] = imputer.fit_transform(data_A_missing)
        print("Impute A successfull")
        imputed_data_B[['ceiling_height_agl:m','cloud_base_agl:m']] = imputer.fit_transform(data_B_missing)
        print("Impute B successfull")
        imputed_data_C[['ceiling_height_agl:m','cloud_base_agl:m']] = imputer.fit_transform(data_C_missing)
        print("Impute C successfull")

        if for_training == True:
            if use_basic_values == True:
                self.training_A = pd.concat([self.training_A, imputed_data_A], axis=1)
                self.training_B = pd.concat([self.training_B, imputed_data_B], axis=1)
                self.training_C = pd.concat([self.training_C, imputed_data_C], axis=1)
            else:
                self.training_A_aggregated_values = pd.concat([self.training_A_aggregated_values,
                                                            imputed_data_A], axis=1)
                self.training_B_aggregated_values = pd.concat([self.training_B_aggregated_values,
                                                            imputed_data_B], axis=1)
                self.training_C_aggregated_values = pd.concat([self.training_C_aggregated_values,
                                                            imputed_data_C], axis=1)
        else:
            self.test_A = pd.concat([self.test_A, imputed_data_A], axis=1)
            self.test_B = pd.concat([self.test_B, imputed_data_B], axis=1)
            self.test_C = pd.concat([self.test_C, imputed_data_C], axis=1)

    def get_training_A(self) -> pd.DataFrame:
        """
        Returns the training data from location A.
        """
        return self.training_A
    
    def get_training_B(self) -> pd.DataFrame:
        """
        Returns the training data from location B.
        """
        return self.training_B
    
    def get_training_C(self) -> pd.DataFrame:
        """
        Returns the training data from location C.
        """
        return self.training_C

    def create_test_data(self, use_mean_values: bool = False,
                        use_median_values: bool = False,
                        use_summed_values: bool = False,
                        impute_data: bool = False) -> pd.DataFrame:
        """
        Returns a DataFrame with the test data from all locations.
        Index is set to 'date_forecast'.
        Feature 'snow_density:kgm3' is dropped because we miss all of the values.

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
        # data_A = ReadData('A')
        # data_B = ReadData('B')
        # data_C = ReadData('C')

        # test_A = data_A.import_test_estimated_data()
        # test_B = data_B.import_test_estimated_data()
        # test_C = data_C.import_test_estimated_data()

        if use_mean_values == True or use_median_values == True or use_summed_values == True:
            self.test_A.set_index('date_forecast', inplace=True)
            self.test_B.set_index('date_forecast', inplace=True)
            self.test_C.set_index('date_forecast', inplace=True)

            if impute_data == True:
                self.impute_data(data_A=self.test_A, data_B=self.test_B, data_C=self.test_C,
                                use_basic_values=False, for_training=False)
            
            if use_mean_values == True:      
                self.test_A = self.test_A.resample('H').mean(numeric_only=False)
                self.test_B = self.test_B.resample('H').mean(numeric_only=False)
                self.test_C = self.test_C.resample('H').mean(numeric_only=False)

                test_data = pd.concat([self.test_A, self.test_B, self.test_C], axis=0)

                test_data.dropna(how='all', axis=0, inplace=True)
            
            elif use_median_values == True:          
                self.test_A = self.test_A.resample('H').median(numeric_only=False)
                self.test_B = self.test_B.resample('H').median(numeric_only=False)
                self.test_C = self.test_C.resample('H').median(numeric_only=False)

                test_data = pd.concat([self.test_A, self.test_B, self.test_C], axis=0)

                test_data.dropna(how='all', axis=0, inplace=True)
            
            elif use_summed_values == True:
                self.test_A = self.test_A.resample('H').sum(numeric_only=True)
                self.test_B = self.test_B.resample('H').sum(numeric_only=True)
                self.test_C = self.test_C.resample('H').sum(numeric_only=True)

                test_data = pd.concat([self.test_A, self.test_B, self.test_C], axis=0)

                test_data.dropna(how='all', axis=0, inplace=True)

                all_zeros = (test_data == 0).all(axis=1)
                test_data = test_data[~all_zeros]
            
        else:
            if impute_data == True:
                self.impute_data(data_A=self.test_A, data_B=self.test_B, data_C=self.test_C,
                                use_basic_values=True, for_training=False)
                
                test_data = pd.concat([self.test_A, self.test_B, self.test_C], axis=0)
                test_data.set_index('date_forecast', inplace=True)

        test_data.drop('snow_density:kgm3', axis=1, inplace=True)

        return test_data
    
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