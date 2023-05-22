import pandas as pd
import numpy as np

from evidently import ColumnMapping

from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import *

from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset
from evidently.tests import *

from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab

from evidently.model_profile import Profile
from evidently.profile_sections import DataDriftProfileSection
from evidently.test_preset import DataDriftTestPreset

from evidently.analyzers.data_drift_analyzer import DataDriftAnalyzer

current = pd.read_csv('./data/current_data.csv')
reference = pd.read_csv('./data/reference_data.csv')

report = Report(
    [
        DataDriftPreset(),
    ]
)

report.run(reference_data=reference, current_data=current)
report.save_html('./src/reports/file.html')