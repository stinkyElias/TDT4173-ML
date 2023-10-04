import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class Data:
    def __init__(self, listOfData) -> None:
        self.A_est = listOfData[0]
        self.A_obs = listOfData[1]
        self.A_test = listOfData[2]
        self.B_est = listOfData[3]
        self.B_obs = listOfData[4]
        self.B_test = listOfData[5]
        self.C_est = listOfData[6]
        self.C_obs = listOfData[7]
        self.C_test = listOfData[8]
        self.A_target = listOfData[9]
        self.B_target = listOfData[10]
        self.C_target = listOfData[11]

    def create_test_data(self):
        # merge X_train_observed and X_train_estimated (without its first column) into one dataframe
        self.A_train = pd.concat([self.A_obs, self.A_est.iloc[:,1:]], axis=0)
        self.B_train = pd.concat([self.B_obs, self.B_est.iloc[:,1:]], axis=0)
        self.C_train = pd.concat([self.C_obs, self.C_est.iloc[:,1:]], axis=0)

        # remove the first column of X_test_estimated
        self.A_test = self.A_test.iloc[:,1:]
        self.B_test = self.B_test.iloc[:,1:]
        self.C_test = self.C_test.iloc[:,1:]

    def add_time_features(self):
        self.A_train = add_time_features(self.A_train)
        self.B_train = add_time_features(self.B_train)
        self.C_train = add_time_features(self.C_train)

        self.A_test = add_time_features(self.A_test)
        self.B_test = add_time_features(self.B_test)
        self.C_test = add_time_features(self.C_test)

    def convert_timestamp_to_datetime(self):
        self.A_train['date_forecast'] = self.A_train['date_forecast'].apply(convert_timestamp_to_datetime)
        self.A_train.set_index('date_forecast', inplace=True)
        self.B_train['date_forecast'] = self.B_train['date_forecast'].apply(convert_timestamp_to_datetime)
        self.B_train.set_index('date_forecast', inplace=True)
        self.C_train['date_forecast'] = self.C_train['date_forecast'].apply(convert_timestamp_to_datetime)
        self.C_train.set_index('date_forecast', inplace=True)

        self.A_test['date_forecast'] = self.A_test['date_forecast'].apply(convert_timestamp_to_datetime)
        self.A_test.set_index('date_forecast', inplace=True)
        self.B_test['date_forecast'] = self.B_test['date_forecast'].apply(convert_timestamp_to_datetime)
        self.B_test.set_index('date_forecast', inplace=True)
        self.C_test['date_forecast'] = self.C_test['date_forecast'].apply(convert_timestamp_to_datetime)
        self.C_test.set_index('date_forecast', inplace=True)

    def add_time_features(self):
        self.A_train = add_time_features(self.A_train)
        self.B_train = add_time_features(self.B_train)
        self.C_train = add_time_features(self.C_train)

        self.A_test = add_time_features(self.A_test)
        self.B_test = add_time_features(self.B_test)
        self.C_test = add_time_features(self.C_test)

        self.A_est = add_time_features(self.A_est)
        self.B_est = add_time_features(self.B_est)
        self.C_est = add_time_features(self.C_est)

    def add_building_feature(self):
        self.A_train['building'] = 0
        self.A_test['building'] = 0
        self.A_target['building'] = 0
        self.B_train['building'] = 1
        self.B_test['building'] = 1
        self.B_target['building'] = 1
        self.C_train['building'] = 2
        self.C_test['building'] = 2
        self.C_target['building'] = 2

        # add the target column to the training data
        self.A_train['pv_measurement'] = self.A_target['pv_measurement']
        self.B_train['pv_measurement'] = self.B_target['pv_measurement']
        self.C_train['pv_measurement'] = self.C_target['pv_measurement']

    def create_train_data(self, custom_features=None):
        train = pd.concat([self.A_train, self.B_train, self.C_train], axis=0)
        test = pd.concat([self.A_test, self.B_test, self.C_test], axis=0)
        target = pd.concat([self.A_target, self.B_target, self.C_target], axis=0)

        # slice test so that it ranges from May 1st to July 5th 2023
        test = test.loc['2023-05-01':'2023-07-05']

        # sort the dataframes by date
        train.sort_index(inplace=True)
        test.sort_index(inplace=True)
        target.sort_index(inplace=True)


        if custom_features is None:
            # create a features list containing all the features in the top row of A_train
            FEATURES = list(train.columns)
            FEATURES.remove('pv_measurement')
            # FEATURES = ['hour', 'day_of_week', 'quarter', 'month', 'year', 'cloud_base_agl:m']
            TARGET = ['pv_measurement']
        else:
            FEATURES = custom_features
            TARGET = ['pv_measurement']

        X_train = train[FEATURES]
        y_train = train[TARGET]

        # In both X_train and y_train, we have to remove the rows where pv_measurement is NaN but keep building
        X_train = X_train[~y_train['pv_measurement'].isna()]
        y_train = y_train[~y_train['pv_measurement'].isna()]

        return X_train, y_train

def read_data() -> [pd.DataFrame]:
    A_est = pd.read_parquet('data/A/parquet/X_train_estimated.parquet', engine='pyarrow')
    A_obs = pd.read_parquet('data/A/parquet/X_train_observed.parquet', engine='pyarrow')
    A_test = pd.read_parquet('data/A/parquet/X_test_estimated.parquet', engine='pyarrow')
    B_est = pd.read_parquet('data/B/parquet/X_train_estimated.parquet', engine='pyarrow')
    B_obs = pd.read_parquet('data/B/parquet/X_train_observed.parquet', engine='pyarrow')
    B_test = pd.read_parquet('data/B/parquet/X_test_estimated.parquet', engine='pyarrow')
    C_est = pd.read_parquet('data/C/parquet/X_train_estimated.parquet', engine='pyarrow')
    C_obs = pd.read_parquet('data/C/parquet/X_train_observed.parquet', engine='pyarrow')
    C_test = pd.read_parquet('data/C/parquet/X_test_estimated.parquet', engine='pyarrow')

    A_target = pd.read_parquet('data/A/parquet/train_targets.parquet', engine='pyarrow')
    B_target = pd.read_parquet('data/B/parquet/train_targets.parquet', engine='pyarrow')
    C_target = pd.read_parquet('data/C/parquet/train_targets.parquet', engine='pyarrow')

    return [A_est, A_obs, A_test, B_est, B_obs, B_test, C_est, C_obs, C_test, A_target, B_target, C_target]

def add_time_features(df) -> pd.DataFrame:
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.day_of_week
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['day_of_year'] = df.index.dayofyear
    df['day_of_month'] = df.index.day
    df['week_of_year'] = df.index.weekofyear
    return df

def convert_timestamp_to_datetime(timestamp):
    try:
        # Convert Timestamp to Python datetime
        datetime_object = timestamp.to_pydatetime()
        return datetime_object
    except AttributeError as e:
        print(f"Error: {e}")
        return None
    
def add_time_features(df):
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.day_of_week
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['day_of_year'] = df.index.dayofyear
    df['day_of_month'] = df.index.day
    df['week_of_year'] = df.index.weekofyear


