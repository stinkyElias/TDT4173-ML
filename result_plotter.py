import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('results/h2o_featurewiz_minus_snow.csv')

# Plot the data with seaborn
sns.set_style("whitegrid")
sns.set_context("paper")
sns.set(font_scale=1.5)
fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(x='id', y='prediction', data=data, ax=ax)
ax.set_xlabel('id')
ax.set_ylabel('pv_measurement')
plt.tight_layout()

plt.show()