import os
import pandas as pd
import matplotlib.pyplot as plt

# get script path
script_directory = os.path.dirname(os.path.abspath(__file__))

# Load observed data
X_train_observed_a = pd.read_csv(os.path.join(script_directory, '../main_project/data/A/csv/X_train_observed.csv'))
X_train_observed_b = pd.read_csv(os.path.join(script_directory, '../main_project/data/B/csv/X_train_observed.csv'))
X_train_observed_c = pd.read_csv(os.path.join(script_directory, '../main_project/data/C/csv/X_train_observed.csv'))

# Features to plot
feature_name_1 = 'diffuse_rad:W'
feature_name_2 = 'direct_rad:W'

# Plots
fig1, axs1 = plt.subplots(3, 1, figsize=(20, 10), sharex=True)
fig2, axs2 = plt.subplots(3, 1, figsize=(20, 10), sharex=True)

X_train_observed_a[['date_forecast', feature_name_1]].set_index('date_forecast').plot(ax=axs1[0], title='Observed Diffuse Radiation - A', color='red')
X_train_observed_b[['date_forecast', feature_name_1]].set_index('date_forecast').plot(ax=axs1[1], title='Observed Diffuse Radiation - B', color='green')
X_train_observed_c[['date_forecast', feature_name_1]].set_index('date_forecast').plot(ax=axs1[2], title='Observed Diffuse Radiation - C', color='blue')

X_train_observed_a[['date_forecast', feature_name_2]].set_index('date_forecast').plot(ax=axs2[0], title='Observed Direct Radiation - A', color='red')
X_train_observed_b[['date_forecast', feature_name_2]].set_index('date_forecast').plot(ax=axs2[1], title='Observed Direct Radiation - B', color='green')
X_train_observed_c[['date_forecast', feature_name_2]].set_index('date_forecast').plot(ax=axs2[2], title='Observed Direct Radiation - C', color='blue')

# Add labels on x- and y-axis
x_label = 'Time and date'
y_label1 = 'Diffuse Radiation Flux [W/m^2]'
y_label2 = 'Direct Radiation Flux [W/m^2]'

for i in range(3):
    axs1[i].set_ylabel(y_label1)
    axs2[i].set_ylabel(y_label2)

axs1[2].set_xlabel(x_label)
axs2[2].set_xlabel(x_label)

plt.show()