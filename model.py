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

WAGE_URL = "https://raw.githubusercontent.com/intro-stat-learning/ISLP/main/ISLP/data/Wage.csv"
OUTPUT_FOLDER = tempfile.gettempdir()
FILE_NAME = "wage.csv"

def load_wage():
    full_path = os.path.join(OUTPUT_FOLDER, FILE_NAME)

    if os.path.exists(full_path):
        df = pd.read_csv(full_path)
    else:
        try:
            urllib.request.urlretrieve(WAGE_URL, full_path)
        except Exception as e:
            print(e)
            raise

        df = pd.read_csv(full_path)

    return df


def describe_wage(df):
    d = {}
    d['n'] = df.shape[0]
    d['columns'] = df.shape[1]
    d['age_range'] = (int(df['age'].min()), int(df['age'].max()))
    d['wage_mean'] = round(df['wage'].mean(), 2)
    d['n_education_levels'] = df['education'].nunique()

    return d

# Step 2 - split_wage
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split_wage(df, test_size=0.25, random_state=0):
    # TODO: (train, test) DataFrames
    df_train, df_test = train_test_split(df, test_size = test_size, random_state = random_state)
    return (df_train, df_test)

def age_xy(df):
    # TODO: (df[['age']], df['wage'])
    X, y = df[['age']], df['wage']
    return (X, y)

def age_grid(lo=18, hi=80, n=63):
    # TODO: one-column DataFrame 'age' over linspace(lo, hi, n)
    df = pd.DataFrame(np.linspace(lo, hi, n), columns = ['age'])
    return df

# Step 3 - cv_tools
import numpy as np
from sklearn.model_selection import cross_val_score


def cv_mse(model, X, y, cv):
    scores = cross_val_score(model, X, y, cv=cv, scoring = 'neg_mean_squared_error')
    mean_score = -scores.mean()
    std_score = scores.std()
    return (mean_score, std_score)

def cv_curve(make_model, X, y, values, cv):
    means = []
    stds = []

    for value in values:
        model = make_model(value)
        mean, std = cv_mse(model, X, y, cv)
        means.append(mean)
        stds.append(std)
    
    return means, stds

def one_se_rule(values, means, ses, prefer='smaller'):
    pref_values = []
    pref_mean = min(means)
    for i in range(len(ses)):
        if(means[i] == pref_mean):
            pref_se = ses[i]
    
    lb = pref_mean - pref_se
    ub = pref_mean + pref_se

    for i in range(len(means)):
        if(means[i] >= lb and means[i]<=ub):
            pref_values.append(values[i])
    if(prefer == "smaller"):
        return min(pref_values)
    else:
        return max(pref_values)

# Step 4 - polynomial_regression
import numpy as np
from scipy.stats import f as f_dist
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

def poly_model(degree):
    # TODO: StandardScaler -> PolynomialFeatures(degree, include_bias=False) -> LinearRegression
    scaler = StandardScaler()
    features = PolynomialFeatures(degree, include_bias = False)
    model = LinearRegression()
    return make_pipeline(scaler, features, model)

def poly_curve(X, y, degrees, cv):
    # TODO: cv_curve over degrees
    means, stds = cv_curve(poly_model, X, y, degrees, cv)
    return (means, stds)

def anova_degrees(X, y, max_degree):
    # TODO: [(d, F, p)] for the nested test of degree d against d - 1, d = 2..max_degree
    sol = []
    n = len(X)
    for d in range(2, max_degree+1):
        model_d = poly_model(d)
        model_d_1 = poly_model(d-1)
        model_d.fit(X, y)
        model_d_1.fit(X, y)

        pred_d = np.array(model_d.predict(X))
        pred_d_1 = np.array(model_d_1.predict(X))

        y = np.array(y)

        rss_d = np.sum((y-pred_d)**2)
        rss_d_1 = np.sum((y-pred_d_1)**2)

        f = ((rss_d_1 - rss_d)*(n-d-1))/rss_d
        p = f_dist.sf(f, 1, n-d-1)

        sol.append([d, round(f, 2), round(p, 4)])
    
    return sol

def choose_degree(X, y, degrees, cv, alpha=0.05):
    # TODO: (CV minimizer over degrees, ANOVA choice: last significant degree before the first failure)
    # selecting the degree with min error
    means, _ = poly_curve(X, y, degrees, cv)
    best_degree = degrees[np.argmin(means)]

    max_degree = max(degrees)
    anova_choice = -1
    sol = anova_degrees(X, y, max_degree)

    for d, f, p in sol:
        if(p>=alpha):
            anova_choice = d-1
            break
        
    return (best_degree, anova_choice)

def curve_on_grid(model, grid):
    # TODO: 1-D predictions on the grid rounded to 2 decimals
    grid = np.asarray(grid).reshape(-1, 1)
    pred = model.predict(grid)
    return np.round(pred, 2)

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

