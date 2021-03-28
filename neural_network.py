import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import explained_variance_score, \
    mean_absolute_error, \
    median_absolute_error
from sklearn.model_selection import train_test_split

import main_2
import constants

day_N_meantempm_string = 'day_{}_meantempm'.format(constants.N_DAY)

df = main_2.get_df()

predictors = ['meantempm', 'mintempm', 'meandewptm', 'maxdewptm', 'mindewptm', 'maxtempm',
    'meantempm_1',  'meantempm_2',  'meantempm_3', 
    'mintempm_1',   'mintempm_2',   'mintempm_3',
    'meandewptm_1', 'meandewptm_2', 'meandewptm_3',
    'maxdewptm_1',  'maxdewptm_2',  'maxdewptm_3',
    'mindewptm_1',  'mindewptm_2',  'mindewptm_3',
    'maxtempm_1',   'maxtempm_2',   'maxtempm_3']

df2 = df[[day_N_meantempm_string] + predictors]

# separate the predictor variables (Xs) from the outcome variable (y)
X = df2[predictors]
y = df2['day_{}_meantempm'.format(constants.N_DAY)]

# split data into training set and a temporary set using sklearn.model_selection.traing_test_split
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, random_state=23)

# take the remaining 20% of data in X_tmp, y_tmp and split them evenly
X_test, X_val, y_test, y_val = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=23)

X_train.shape, X_test.shape, X_val.shape
print("Training instances   {}, Training features   {}".format(X_train.shape[0], X_train.shape[1]))
print("Validation instances {}, Validation features {}".format(X_val.shape[0], X_val.shape[1]))
print("Testing instances    {}, Testing features    {}".format(X_test.shape[0], X_test.shape[1]))