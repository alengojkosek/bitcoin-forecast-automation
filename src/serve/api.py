import os
import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
from dagshub.streaming import DagsHubFilesystem
fs = DagsHubFilesystem()

if __name__ == '__main__':
    app.run(debug=True)
