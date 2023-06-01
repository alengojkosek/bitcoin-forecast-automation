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

MLFLOW_TRACKING_URI = (
    "https://dagshub.com/alengojkosek/bitcoin-forecast-automation.mlflow"
)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
os.environ["MLFLOW_TRACKING_USERNAME"] = "alengojkosek"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "af9dc140062b507e1c608237487e976c3d8e7d78"
experiment = mlflow.set_experiment("btc")

# Get Experiment Details
print("Experiment_id: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Tags: {}".format(experiment.tags))
print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

mlflow.autolog(exclusive=False)

with mlflow.start_run():
    df = pd.read_csv("data/current_data.csv")

    df["Date"] = pd.to_datetime(df["Date"])

    features = ["Date", "Close"]
    data = df[features]

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

    time_window = 10

    def create_sliding_window(data, window_size):
        X = []
        y = []
        for i in range(len(data) - window_size):
            X.append(data[i : i + window_size])
            y.append(data[i + window_size])
        return np.array(X), np.array(y)

    X, y = create_sliding_window(scaled_data, time_window)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    model = Sequential()
    model.add(
            LSTM(
                128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])
            )
        )
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(loss="mean_squared_error", optimizer="adam")

    early_stopping = EarlyStopping(
            patience=3, restore_best_weights=True, monitor="loss"
        )

    model.fit(
            X_train,
            y_train,
            epochs=200,
            batch_size=64,
            callbacks=[early_stopping],
            verbose=1,
        )

    y_pred = model.predict(X_test)

    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MAE:", mae)
    print("MSE:", mse)
    print("R2 Score:", r2)

    import pandas as pd

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Create a dictionary to represent the data
    metrike = {
        "Metric": ["MAE", "MSE", "R2 Score"],
        "Value": [mae, mse, r2]
    }

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(metrike)

    # Define the path to the CSV file
    csv_file = "data/metrics.csv"

    # Save the DataFrame to CSV
    df.to_csv(csv_file, index=False)

    print("Metrics saved to", csv_file)

    
    prediction_horizon = 8
    last_date = df['Date'].iloc[0]  # Get the last date from your raw_data.csv
    next_dates = pd.date_range(start=last_date, periods=prediction_horizon)  # Generate the next 7 dates

    # Scale the last known close price
    last_close = data['Close'].iloc[0]
    scaled_last_close = scaler.transform([[last_close]])

    # Prepare the input sequence
    input_sequence = np.zeros((prediction_horizon, time_window, 1))
    input_sequence[0] = X[0]  # Last sequence from the test data

    # Predict the next 7 days
    predicted_prices = []

    for i in range(prediction_horizon):
        if i > 0:
            input_seq = np.reshape(input_sequence[i-1], (1, input_sequence[i-1].shape[0], input_sequence[i-1].shape[1]))

            # Predict the next time step
            predicted_price = model.predict(input_seq)

            # Append the predicted value to the input sequence
            input_sequence[i] = np.concatenate([input_sequence[i-1, 1:], predicted_price])

            # Reverse scaling for the predicted price
            predicted_price = scaler.inverse_transform(predicted_price)

            # Append the predicted price to the list of predictions
            predicted_prices.append(predicted_price[0][0])


            # Print the predicted price with date if within the range
            if i < len(next_dates) and i < len(predicted_prices):
                print(f"{next_dates[i].date()}: {predicted_prices[i]}")

    # Create a DataFrame with the predicted prices and dates
    predictions_df = pd.DataFrame({'Date': next_dates[:len(predicted_prices)], 'Price': predicted_prices})

    # Save the DataFrame to CSV
    predictions_df.to_csv("data/predictions/future_data.csv", index=False)

    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("R2 score", r2)
mlflow.end_run()
