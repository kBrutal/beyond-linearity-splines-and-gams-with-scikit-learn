"""
Beyond Linearity: Splines and GAMs with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_wage
import os
import tempfile
import urllib.request
import pandas as pd
import requests

WAGE_URL = "https://raw.githubusercontent.com/intro-stat-learning/ISLP/main/ISLP/data/Wage.csv"
OUTPUT_FOLDER = tempfile.gettempdir()
FILE_NAME = "wage.csv"

def load_wage():
    full_path = os.path.join(OUTPUT_FOLDER, FILE_NAME)
    if os.path.exists(full_path):
        df = pd.read_csv(full_path)
    else:
        try:
            response = requests.get(WAGE_URL, timeout = 10)
            if response.status_code == 200:
                with open(full_path, "wb") as f:
                    f.write(response.content)
        except Exception as e:
            print(e)
        df = pd.read_csv(full_path)
    return df

def describe_wage(df):
    # TODO: {'n', 'columns', 'age_range', 'wage_mean' rounded 2, 'n_education_levels'}
    d = {}
    full_path = os.path.join(OUTPUT_FOLDER, FILE_NAME)
    df = pd.read_csv(full_path)
    d['n'] = df.shape[0]
    d['columns'] = df.shape[1]
    d['age_range'] = (int(df['age'].min()), int(df['age'].max()))
    d['wage_mean'] = round(df['wage'].mean(), 2)
    d['n_education_levels'] = df['education'].nunique()

    return d

# Step 2 - split_wage (not yet solved)
# TODO: implement

# Step 3 - cv_tools (not yet solved)
# TODO: implement

# Step 4 - polynomial_regression (not yet solved)
# TODO: implement

# Step 5 - step_functions (not yet solved)
# TODO: implement

# Step 6 - spline_regression (not yet solved)
# TODO: implement

# Step 7 - extrapolation (not yet solved)
# TODO: implement

# Step 8 - smoothing_spline (not yet solved)
# TODO: implement

# Step 9 - local_smoother (not yet solved)
# TODO: implement

# Step 10 - gam_pipeline (not yet solved)
# TODO: implement

# Step 11 - partial_effects (not yet solved)
# TODO: implement

# Step 12 - logistic_gam (not yet solved)
# TODO: implement

# Step 13 - fit_age_models (not yet solved)
# TODO: implement

# Step 14 - test_comparison (not yet solved)
# TODO: implement

