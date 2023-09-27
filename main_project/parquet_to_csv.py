import os
import pandas as pd

# Get script path
script_directory = os.path.dirname(os.path.abspath(__file__))

# Load training data
train_a = pd.read_parquet(os.path.join(script_directory, '../main_project/data/A/parquet/train_targets.parquet'))
train_b = pd.read_parquet(os.path.join(script_directory, '../main_project/data/B/parquet/train_targets.parquet'))
train_c = pd.read_parquet(os.path.join(script_directory, '../main_project/data/C/parquet/train_targets.parquet'))

# Load test estimation data
X_test_estimated_a = pd.read_parquet(os.path.join(script_directory, '../main_project/data/A/parquet/X_test_estimated.parquet'))
X_test_estimated_b = pd.read_parquet(os.path.join(script_directory, '../main_project/data/B/parquet/X_test_estimated.parquet'))
X_test_estimated_c = pd.read_parquet(os.path.join(script_directory, '../main_project/data/C/parquet/X_test_estimated.parquet'))

# Load training estimation data
X_train_estimated_a = pd.read_parquet(os.path.join(script_directory, '../main_project/data/A/parquet/X_train_estimated.parquet'))
X_train_estimated_b = pd.read_parquet(os.path.join(script_directory, '../main_project/data/B/parquet/X_train_estimated.parquet'))
X_train_estimated_c = pd.read_parquet(os.path.join(script_directory, '../main_project/data/C/parquet/X_train_estimated.parquet'))

# Load training observation data
X_train_observed_a = pd.read_parquet(os.path.join(script_directory, '../main_project/data/A/parquet/X_train_observed.parquet'))
X_train_observed_b = pd.read_parquet(os.path.join(script_directory, '../main_project/data/B/parquet/X_train_observed.parquet'))
X_train_observed_c = pd.read_parquet(os.path.join(script_directory, '../main_project/data/C/parquet/X_train_observed.parquet'))

# Export DataFrames to csv for visualization
train_a.to_csv(os.path.join(script_directory, '../main_project/data/A/csv/train_targets.csv'), index=False)
train_b.to_csv(os.path.join(script_directory, '../main_project/data/B/csv/train_targets.csv'), index=False)
train_c.to_csv(os.path.join(script_directory, '../main_project/data/C/csv/train_targets.csv'), index=False)

X_test_estimated_a.to_csv(os.path.join(script_directory, '../main_project/data/A/csv/X_test_estimated.csv'), index=False)
X_test_estimated_b.to_csv(os.path.join(script_directory, '../main_project/data/B/csv/X_test_estimated.csv'), index=False)
X_test_estimated_c.to_csv(os.path.join(script_directory, '../main_project/data/C/csv/X_test_estimated.csv'), index=False)

X_train_estimated_a.to_csv(os.path.join(script_directory, '../main_project/data/A/csv/X_train_estimated.csv'), index=False)
X_train_estimated_b.to_csv(os.path.join(script_directory, '../main_project/data/B/csv/X_train_estimated.csv'), index=False)
X_train_estimated_c.to_csv(os.path.join(script_directory, '../main_project/data/C/csv/X_train_estimated.csv'), index=False)

X_train_observed_a.to_csv(os.path.join(script_directory, '../main_project/data/A/csv/X_train_observed.csv'), index=False)
X_train_observed_b.to_csv(os.path.join(script_directory, '../main_project/data/B/csv/X_train_observed.csv'), index=False)
X_train_observed_c.to_csv(os.path.join(script_directory, '../main_project/data/C/csv/X_train_observed.csv'), index=False)