
import mlflow
import os
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

    features = ['Date', 'Close']  # Only keeping Date and Close as features
    data = df[features]

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    time_window = 10

    def create_sliding_window(data, window_size):
        X = []
        y = []
        for i in range(len(data) - window_size):
            X.append(data[i:i+window_size])
            y.append(data[i+window_size])  # Appending the Close value
        return np.array(X), np.array(y)

    X, y = create_sliding_window(scaled_data, time_window)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, shuffle=False)

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    early_stopping = EarlyStopping(patience=3, restore_best_weights=True, monitor='loss')

    model.fit(X_train, y_train, epochs=20, batch_size=64, callbacks=[early_stopping], verbose=1)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Reverse scaling for the predicted and actual values
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test)

    # Calculate the metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print the metrics
    print("MAE:", mae)
    print("MSE:", mse)
    print("R2 Score:", r2)

    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("R2 score", r2)
mlflow.end_run()