import os
import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['GET'])
def predict():
    file_path = os.path.join('data', 'predictions', 'future_data.csv')
    df = pd.read_csv(file_path)
    return jsonify(df.to_dict(orient='records'))

@app.route('/current')
def current():
    data_path = os.path.join(os.getcwd(), 'data', 'current_data.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        return jsonify(df.to_dict(orient='records'))
    else:
        return jsonify({'error': 'Data file not found'})

@app.route('/metrics')
def metrics():
    data_path = os.path.join(os.getcwd(), 'data', 'metrics.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        return jsonify(df.to_dict(orient='records'))
    else:
        return jsonify({'error': 'Data file not found'})

if __name__ == '__main__':
    app.run(debug=True)
