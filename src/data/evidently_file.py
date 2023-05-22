import pandas as pd
import numpy as np
import evidently
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
current = pd.read_csv('./data/current_data.csv')
reference = pd.read_csv('./data/reference_data.csv')

report = Report(
    [
        DataDriftPreset(),
    ]
)

report.run(reference_data=reference, current_data=current)
report.save_html('./src/reports/file.html')