import os
import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS
from dagshub.streaming import install_hooks

app = Flask(__name__)
CORS(app)

if __name__ == '__main__':
    install_hooks()
    data_path = "bitcoin-forecast-automation/data/predictions/future_data.csv"
    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
        print(data)
    else:
        print("Data file not found.")
    app.run(debug=True)
