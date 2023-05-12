from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import mlflow
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import timedelta, datetime


MLFLOW_TRACKING_URI='https://dagshub.com/alengojkosek/bitcoin-forecast-automation.mlflow'
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
os.environ['MLFLOW_TRACKING_USERNAME'] = 'alengojkosek'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'af9dc140062b507e1c608237487e976c3d8e7d78'
experiment = mlflow.set_experiment("btc")

# Get Experiment Details
print("Experiment_id: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Tags: {}".format(experiment.tags))
print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

mlflow.autolog(exclusive=False)

with mlflow.start_run():

    df = pd.read_csv('data/raw/raw_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])

    # Get the last date from the DataFrame
    last_date = df['Date'].max()

    # Generate 7 future dates starting from the day after the last date
    future_dates = pd.date_range(last_date + timedelta(days=1), periods=7, freq='D')

    # Create a new DataFrame with the future dates
    future_df = pd.DataFrame({'Date': future_dates})

    # Convert the future dates to timestamps and predict the Close values using Linear Regression
    X = df[['Date']].astype(np.int64) // 10**9
    y = df['Close']
    lr = LinearRegression().fit(X, y)

    # Reshape the future dates array to a 2D array with a single feature
    future_dates_2d = future_df['Date'].astype(np.int64) // 10**9
    future_dates_2d = future_dates_2d.values.reshape(-1, 1)

    # Predict the Close values for the future dates
    future_df['Close'] = lr.predict(future_dates_2d)

    # Convert the predicted Close values back to floats
    future_df['Close'] = future_df['Close'].astype(float)

    # Save the future DataFrame to a new CSV file without the index column
    future_df.to_csv('future_data.csv', index=False)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Fit a linear regression model to the training data
    lr = LinearRegression().fit(X_train, y_train)

    # Predict the Close values for the test data
    y_pred = lr.predict(X_test)

    # Calculate the evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MAE:", mae)
    print("MSE:", mse)
    print("R2 Score:", r2)


    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("R2 score", r2)
mlflow.end_run()