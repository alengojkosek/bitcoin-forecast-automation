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

if __name__ == '__main__':
    app.run(debug=True)
