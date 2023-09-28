import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class FeaturePlottingFromCSV:
    def __init__(self, feature_names: list, data_path: str) -> None:       
        self.feature_names = feature_names
        self.data_path = data_path
    
    def plot_data(self) -> None:
        data_a = self.load_combined_data('A')
        data_b = self.load_combined_data('B')
        data_c = self.load_combined_data('C')
        
        sns.set(style='whitegrid')
        
        num_of_features = len(self.feature_names)
        figs = [plt.subplots(3, 1, figsize=(20, 10), sharex=True) for _ in range(num_of_features)]
        
        for i, feature_name in enumerate(self.feature_names):
            figs, axs = figs[i]
            self.plot_subplot(axs, data_a, f'Observed {feature_name} - A', feature_name, 'red')
            self.plot_subplot(axs, data_b, f'Observed {feature_name} - B', feature_name, 'green')
            self.plot_subplot(axs, data_c, f'Observed {feature_name} - C', feature_name, 'blue')
            
            self.set_labels(axs, 'Time and date', f'{feature_name}')
            
        plt.show()
    
    def load_combined_data(self, subdir: str) -> pd.DataFrame:
        data_files = ['X_train_observed.csv', 'X_train_estimated.csv', 'X_test_estimated.csv']
        combined_data = self.load_data(subdir, data_files[0])
        
        for file in data_files[1:]:
            data_to_merge = self.load_data(subdir, file)
            combined_data = pd.merge(combined_data, data_to_merge, on='date_forecast')
        
        return combined_data
    
    def load_data(self, subdir: str, filename: str) -> pd.DataFrame:
        data_file_path = os.path.join('data', subdir, 'csv', filename)
        
        return pd.read_csv(data_file_path)
    
    def plot_subplot(self, axs, data, title, column, color) -> None:
        sns.lineplot(data=data, x='date_forecast', y=column, ax=axs, label=title, color=color)
        # axs.set_title(title)
    
    def set_labels(self, axs, x_label: str, y_label: str) -> None:
        for ax in axs:
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label) # Endre denne
    
    
feature_names = ['diffuse_rad:W', 'direct_rad:W']
data_path = '../data/'
feature_plotter = FeaturePlottingFromCSV(feature_names, data_path)
feature_plotter.plot_data()

# # Load observed data
# X_train_observed_a = pd.read_csv(os.path.join(script_directory, '../main_project/data/A/csv/X_train_observed.csv'))
# X_train_observed_b = pd.read_csv(os.path.join(script_directory, '../main_project/data/B/csv/X_train_observed.csv'))
# X_train_observed_c = pd.read_csv(os.path.join(script_directory, '../main_project/data/C/csv/X_train_observed.csv'))

# # Features to plot
# feature_name_1 = 'diffuse_rad:W'
# feature_name_2 = 'direct_rad:W'

# # Plots
# fig1, axs1 = plt.subplots(3, 1, figsize=(20, 10), sharex=True)
# fig2, axs2 = plt.subplots(3, 1, figsize=(20, 10), sharex=True)

# X_train_observed_a[['date_forecast', feature_name_1]].set_index('date_forecast').plot(ax=axs1[0], title='Observed Diffuse Radiation - A', color='red')
# X_train_observed_b[['date_forecast', feature_name_1]].set_index('date_forecast').plot(ax=axs1[1], title='Observed Diffuse Radiation - B', color='green')
# X_train_observed_c[['date_forecast', feature_name_1]].set_index('date_forecast').plot(ax=axs1[2], title='Observed Diffuse Radiation - C', color='blue')

# X_train_observed_a[['date_forecast', feature_name_2]].set_index('date_forecast').plot(ax=axs2[0], title='Observed Direct Radiation - A', color='red')
# X_train_observed_b[['date_forecast', feature_name_2]].set_index('date_forecast').plot(ax=axs2[1], title='Observed Direct Radiation - B', color='green')
# X_train_observed_c[['date_forecast', feature_name_2]].set_index('date_forecast').plot(ax=axs2[2], title='Observed Direct Radiation - C', color='blue')

# # Add labels on x- and y-axis
# x_label = 'Time and date'
# y_label1 = 'Diffuse Radiation Flux [W/m^2]'
# y_label2 = 'Direct Radiation Flux [W/m^2]'

# for i in range(3):
#     axs1[i].set_ylabel(y_label1)
#     axs2[i].set_ylabel(y_label2)

# axs1[2].set_xlabel(x_label)
# axs2[2].set_xlabel(x_label)

# plt.show()