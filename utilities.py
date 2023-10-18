import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class DataProcessor:
    def __init__(self, data_list):
        (self.A_est, self.A_obs, self.A_test,
        self.B_est, self.B_obs, self.B_test,
        self.C_est, self.C_obs, self.C_test,
        self.A_target, self.B_target, self.C_target) = data_list[:12]

        self.A_train, self.B_train, self.C_train = None, None, None

    def create_test_data(self):
        test = pd.concat([self.A_test, self.B_test, self.C_test], axis=0)
        test.sort_index(inplace=True)
        test.drop(columns=['date_calc'], inplace=True)
        return test

    def create_train_data(self):

        # concat the estimated (without the date_calc column) and observed dataframes
        self.A_train = pd.concat([self.A_obs, self.A_est.iloc[:, 1:]], axis=0)
        self.B_train = pd.concat([self.B_obs, self.B_est.iloc[:, 1:]], axis=0)
        self.C_train = pd.concat([self.C_obs, self.C_est.iloc[:, 1:]], axis=0)

        self._format_time_index()
        self._add_time_features()
        self._add_building_feature()

        # add the target column to the training data
        self.A_train['pv_measurement'] = self.A_target['pv_measurement']
        self.B_train['pv_measurement'] = self.B_target['pv_measurement']
        self.C_train['pv_measurement'] = self.C_target['pv_measurement']

        df = pd.concat([self.A_train, self.B_train, self.C_train], axis=0)

        # for df in [train, target]:
        df.sort_index(inplace=True)

        # # slive it so that it only contains whole hours
        # train = train[train.index.hour == 0]
        # target = target[target.index.hour == 0]

        print(df.shape)


        return df

    def _format_time_index(self):
        # convert the date_forecast column to a datetime object and set it as the index
        for df in [self.A_train, self.B_train, self.C_train, self.A_test, self.B_test, self.C_test]:
            df['date_forecast'] = pd.to_datetime(df['date_forecast'])
            df.set_index('date_forecast', inplace=True)

        # convert the time column to a datetime object and set it as the index
        for df in [self.A_target, self.B_target, self.C_target]:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)

    def _add_time_features(self):
            for df in [self.A_train, self.B_train, self.C_train, self.A_test, self.B_test, self.C_test, self.A_target, self.B_target, self.C_target]:
                df['hour'] = df.index.hour
                df['day_of_week'] = df.index.day_of_week
                df['quarter'] = df.index.quarter
                df['month'] = df.index.month
                df['year'] = df.index.year
                df['day_of_year'] = df.index.dayofyear
                df['day_of_month'] = df.index.day
                df['minute'] = df.index.minute
                df['day'] = df.index.day    

    def _add_building_feature(self):
        for df, building_label in zip(
            [self.A_train, self.B_train, self.C_train, self.A_test, self.B_test, self.C_test, self.A_target, self.B_target, self.C_target],
            [0, 1, 2, 0, 1, 2, 0, 1, 2]
        ):
            df['building'] = building_label


def read_data() -> list:
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

# make a function of this
def save_to_csv(results, filename):
    # slice the results dataframe so that it contains rows where minute is 0
    results = results[results.index.minute == 0]
    results = results[['pv_prediction']]
    results.rename(columns={'pv_prediction': 'prediction'}, inplace=True)
    #reset the index
    results.reset_index(inplace=True)
    #emove date_forecast column
    results.drop(columns=['date_forecast'], inplace=True)
    #rename the index column to "id"
    results['id'] = results.index
    #move the id column to the first column
    cols = list(results.columns)
    cols = [cols[-1]] + cols[:-1]
    results = results[cols]
    #save to csv
    results.to_csv(f'results/{filename}.csv', index=False)
    # print(results.head())

def create_results_dataframe(test, y_pred):
    results = test.copy()
    results = results[['hour', 'day_of_week', 'quarter', 'month', 'year', 'day_of_year', 'day_of_month']]
    res_pd = pd.DataFrame(data=y_pred, index=test.index, columns=['pv_prediction'])
    results["pv_prediction"] = pd.DataFrame(data=res_pd)
    return results

def plot_important_features(model):
    fi = pd.DataFrame(data=model.feature_importances_,
                  index=model.feature_names_in_,
                  columns=['importance'])
    fi.sort_values(by='importance').plot(kind='barh', figsize=(10,15))


def plot_results_daily(train,results):
    fig, ax = plt.subplots(figsize=(12,8))
    sns.lineplot(data=results[results['building'] == 0], x='day_of_month', y='pv_prediction', label='pv_prediction (building A)')
    sns.lineplot(data=results[results['building'] == 1], x='day_of_month', y='pv_prediction', label='pv_prediction (building B)')
    sns.lineplot(data=results[results['building'] == 2], x='day_of_month', y='pv_prediction', label='pv_prediction (building C)')
    sns.lineplot(data=train[train['building'] == 0], x='day_of_month', y='pv_measurement', label='pv_measured (building A) 2021')
    sns.lineplot(data=train[train['building'] == 1], x='day_of_month', y='pv_measurement', label='pv_measured (building B) 2021')
    sns.lineplot(data=train[train['building'] == 2], x='day_of_month', y='pv_measurement', label='pv_measured (building C) 2021')
    ax.set_title('daily pv_prediction and pv_measurement between May 1st and July 5th')
    ax.set_ylabel('pv_measurement')
    ax.set_xlabel('day_of_month')
    ax.legend()