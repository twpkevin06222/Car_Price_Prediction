import numpy as np
import fnmatch
from sklearn.preprocessing import QuantileTransformer
import lightgbm as lgb
import pickle
import math
import re
from collections import deque
from sklearn.metrics import mean_squared_error


def special_char_field_checker(ds, special_char):
    """
    Function to check if special character exist in the
    fields

    @param ds: data set
    @param special_char: special character in regex format
    @return: list of the fields containing special characters
    """
    special_char_field = []
    for cols in ds.columns:
        # iterate for only fields with categorical data
        if ds[cols].dtype != object:
            continue
        special_char_count = ds[cols].str.contains(special_char,
                                                   regex=True).sum()
        if special_char_count == 0:
            continue
        else:
            special_char_field.append(cols)
            print("\'{}\' has {} rows with special characters.".format(cols,
                                                             special_char_count))

    return special_char_field


def string_search(cols_name, wildcard):
    """
    Function to search for fields with wildcard string.

    @param cols_name: List of the dataframe
    @param wildcard: String with wildcard, e.g. "*Code"
    @return: list of indices where the fields contains the stirng
    """
    match_id_list = []
    for i, c in enumerate(cols_name):
        # match the field with wildcard
        if fnmatch.fnmatch(c, wildcard) is True:
            match_id_list.append(i)
    return match_id_list


def check_warranty_km(x):
    """
    Helper function to check if 'Warranty_km' has negative values
    return NAN if exist
    """
    if math.isnan(x):
        return x
    if int(x) < 0:
        return np.nan
    return x


def strip_colors(x):
    """
    Helper function to categorize colour column to common_colours
    given common_colours as wildcard. Else return 'others'
    """
    common_colours = {'gold', 'white', 'silver', 'black', 'grey', 'black',
                     'blue', 'red', 'yellow', 'orange', 'green', 'purple',
                     'brown'}
    patterns = deque([r'\-(.*?)\(', r'-(.*?)'])

    # exhaust pattern search
    while len(patterns) != 0:
        p = patterns.popleft()
        tmp = re.search(p, x)
        if tmp:
            x = tmp.group(1).replace(".", "").replace("(", "").strip()
            break

    for c in common_colours:
        if c in x.lower():
            return c
    return 'others'


def objective(trial, X, Y, cv, folds, model = 'lightgbm'):
    # data scaling
    qt = QuantileTransformer(output_distribution = 'normal')
    if model == 'lightgbm':
        param_grid = {
        "objective": 'regression',
        "verbosity": -1,
        "n_estimators": trial.suggest_categorical("n_estimators", [10000]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        }
        model = lgb.LGBMRegressor(**param_grid)
    cv_scores = np.empty(folds)
    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        x_train, x_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]
        # feature scaling on the fly to prevent data leakage
        qt.fit_transform(x_train)
        qt.fit_transform(x_test)
        qt.fit_transform(y_train.reshape(-1,1))
        qt.fit_transform(y_test.reshape(-1,1))
        model.fit(x_train, y_train,
                 eval_set = [(x_test, y_test)],
                 verbose = False,
                 eval_metric = ['l1','l2'],)
        # prediction
        preds = model.predict(x_test)
        # return MSE
        cv_scores[fold] = mean_squared_error(y_test, preds)
        # save model for each trial to be loaded later
        with open("model_weights/{}.pickle".format(trial.number), "wb") as fout:
            pickle.dump(model, fout)
    return np.mean(cv_scores)

