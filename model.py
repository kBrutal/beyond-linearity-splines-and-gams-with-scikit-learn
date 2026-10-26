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

# Step 5 - step_functions
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer

def step_model(n_bins):
    dis = KBinsDiscretizer(n_bins = n_bins, encode = 'onehot-dense', strategy = 'uniform')
    lr = LinearRegression()
    return make_pipeline(dis, lr)

def step_curve(X, y, bins, cv):
    return cv_curve(step_model, X, y, bins, cv)

def choose_bins(X, y, bins, cv):
    means, stds = step_curve(X, y, bins, cv)
    idx_min = np.argmin(means)
    bins_min = bins[idx_min]
    thershold = means[idx_min] + stds[idx_min]
    for b in bins:
        if(means[b] <= thershold):
            bins_1se = b
            return (bins_min, bins_1se)

def bin_edges(model):
    discretizer = model[0]
    edges = discretizer.bin_edges_[0]
    return [float(x) for x in np.round(edges, 1)]
    

def step_levels(model, grid):
    pred = model.predict(grid)
    return np.sort(np.unique(np.round(pred, 1)))

# Step 6 - spline_regression
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

def spline_model(n_knots, degree=3, extrapolation='constant'):
    st = SplineTransformer(n_knots, degree, extrapolation=extrapolation, knots='quantile', include_bias = False)
    lr = LinearRegression()
    return make_pipeline(st, lr)

def spline_basis_size(model, X):
    model[0].fit(X)
    return model[0].n_features_out_

def spline_curve(X, y, knot_counts, cv):
    return cv_curve(spline_model, X, y, knot_counts, cv)

def choose_knots(X, y, knot_counts, cv):
    means, stds = spline_curve(X, y, knot_counts, cv)
    idx_min = np.argmin(means)
    k_min = knot_counts[idx_min]

    threshold = means[idx_min] + stds[idx_min]

    for i in range(len(means)):
        if(means[i]<=threshold):
            k_1se = knot_counts[i]
            return (k_min, k_1se)

# Step 7 - extrapolation
import numpy as np
import pandas as pd

def beyond_data(models, ages):
    pred = {}
    X = pd.DataFrame({"age": ages})
    for name, model in models.items():
        p = model.predict(X)
        pred[name] = list(np.round(p, 1))
    return pred

def extrapolation_report(X, y, ages):
    # TODO: poly4, spline_const, spline_linear predictions beyond the data, plus 'poly4_range'
    models = {
        'poly4': poly_model(4),
        'spline_const': spline_model(5),
        'spline_linear': spline_model(5, extrapolation = 'linear')
    }

    for model in models.values():
        model.fit(X, y)
    
    report = beyond_data(models, ages)

    poly4_preds = report['poly4']
    report['poly4_range'] = round(max(poly4_preds) - min(poly4_preds), 1)
    return report

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

