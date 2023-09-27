# TDT4173-ML
Repository for the course TDT4173 Machine Learning at NTNU autumn 2023

# Main project assignment task

Build a model that can predict solar energy production for every hour of the next day.
Problem type is time-series forecasting and regression problem.

Three offices that produce energy. For this data set, you have to take into account the
weather and the time of day. History is not enough.

We get 45 weather related features that describe the weather. For example cloud cover, air
density etc.

y_train: energy production (main task to predict!)
x_train_observed: weather related features (45 features in total)
x_train_estimated: real weather

y_test: what we need to predict
x_test_estimated: saved predictions by the weather forecast

Problem #1
We do not know the future weather, but we can use weather predictions.
Every day we receive the weather predictions for the next 24 hours for all 45 features.
The model have to take the forecast into account. "Take prediction and make a new prediction"

Problem #2
How good is the weather forecast? Weather predictions differ from real observed weather.
Our model must adapt to the distribution difference between real weather and estimated weather.

???
We get 4,5 years of energy production, where 4 years is the real weather, and we get half a year
of weather forecasts (x_train_estimated).
y_test is 30 days. Based on the 24 hours weather forecast, we have to estimate 30 times
???

# Evaluation metric
Evaluation metric is Mean Absolute Error. 30 days x 24 hours x 3 locations
MAE = 1/n * sum(abs(y - y_hat))

# Kaggle
Kaggle is split between a public and private leaderboard. We will just see half of our score during
the semester, until after the deadline. Grading is dependent on the private leaderboard.

We can submit only 5 times a day. Before the competition end you have to select two iterations.
Select diverse solutions and choose the best of them. For example, choose two different methods
to solve the problem.

Do not overfit on the public leaderboard score.

Useful steps:
    - EDA (explorative data analysis)
    - Accurate preprocessing
    - Feature engineering
    - Strong models
    - Hyperparameters
    - Stacking
    - Results analysis

 What should be delivered?
    1. Select two predictions on Kaggle before the deadline
    2. Upload two short Jupyter notebooks
        - Only necessary steps to reproduce your selected predictions
        - Naming: "short_notebook_1.ipynb"
        - Put your group number!
    3. One long Jupyter notebook
        - Contains all attempts in your group work (EDA, all models, algorithms, feature engineering,
            results interpretation etc.)
Submission to Kaggle and notebooks to Blackboard.

We cannot use external data other than provded. Writing massive data in code is not allowed. Everything
else is allowed: whatever language, tools, platforms, AutoML (offline), libraries and file formats
during development.

Possible deductions:
    - Pass individual assignment in the second chance (-5 points)
    - Late submission (-10 points)
    - No exploratory data analysis (EDA) (-3 points)
    - Only one predictor used (-3 points)
    - No feature engineering (-3 points)
    - No model interpretation (-3 points)
All deductions are binary - either full or no deduction.

# Computing resources
- NTNU computing resources
- Google Cloud Credits