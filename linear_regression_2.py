from datetime import datetime, timedelta
import time
from collections import namedtuple
import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, median_absolute_error
import main_2
import constants

## Linear Regression ##
df = main_2.get_df()
print(df.info())

# To assess the linearity between our independent variable, which
# for now is the mean temperature, and the other independent
# variables we will calculate the Pearson correlation coefficient
pcc = df.corr()[['meantempm']].sort_values('meantempm')
print(pcc)

day_N_meantempm_string = 'day_{}_meantempm'.format(constants.N_DAY)

# Remove the features that have correlation values less than the absolute value of 0.6
# We do this by creating a new DataFrame that only contains the variables of interest
predictors = ['meantempm', 'mintempm', 'meandewptm', 'maxdewptm', 'mindewptm', 'maxtempm',
    'meantempm_1',  'meantempm_2',  'meantempm_3', 
    'mintempm_1',   'mintempm_2',   'mintempm_3',
    'meandewptm_1', 'meandewptm_2', 'meandewptm_3',
    'maxdewptm_1',  'maxdewptm_2',  'maxdewptm_3',
    'mindewptm_1',  'mindewptm_2',  'mindewptm_3',
    'maxtempm_1',   'maxtempm_2',   'maxtempm_3']
# print("\n legnth = {}\n".format(len(predictors)))
df2 = df[[day_N_meantempm_string] + predictors]

# Visualizing the Relationships:

# manually set the parameters of the figure to and appropriate size
plt.rcParams['figure.figsize'] = [16, 22]

# call subplots specifying the grid structure we desire and that 
# the y axes should be shared
fig, axes = plt.subplots(nrows=6, ncols=4, sharey=True)

# Since it would be nice to loop through the features in to build this plot
# let us rearrange our data into a 2D array of 6 rows and 3 columns
arr = np.array(predictors).reshape(6, 4)

# use enumerate to loop over the arr 2D array of rows and columns
# and create scatter plots of each meantempm vs each feature
for row, col_arr in enumerate(arr):
    for col, feature in enumerate(col_arr):
        axes[row, col].scatter(df2[feature], df2[day_N_meantempm_string])
        if col == 0:
            axes[row, col].set(xlabel=feature, ylabel=day_N_meantempm_string)
        else:
            axes[row, col].set(xlabel=feature)
# plt.show()



# Backward elimination:

# separate the predictor variables (Xs) from the outcome variable (y)
X = df2[predictors]
y = df2['day_{}_meantempm'.format(constants.N_DAY)]

# add a constant to the predictor variable set to represent the Bo intercept
X = sm.add_constant(X)

def get_key(pvalues, val):
	for key, value in pvalues.items():
		if val == value:
			return key

	return "key doesn't exist"

def backward_elimination(X, y, p_value=0.05):
    removed_predictor = {}
    
    # (1) select a significance level (default: 0.05)
    
    while True:
        # (2) Fit the model
        model = sm.OLS(y, X).fit()

        # (3) identify the feature (predictor) which has the highest P-value
        max_pvalue_predictor = max(model.pvalues)
        max_pvalue_predictor_key = get_key(model.pvalues, max_pvalue_predictor)

        # (4) if the max pvalue found is greater than the significance level - remove predictor
        if max_pvalue_predictor > p_value:
            X = X.drop(max_pvalue_predictor_key, axis=1)
            removed_predictor[max_pvalue_predictor_key] = max_pvalue_predictor
            # temp.append(max_pvalue_predictor_key)
        else:
            break
    print("\nPredicators removed by backward elimination:\n", removed_predictor)
    print(model.summary())


backward_elimination(X, y)

# Linear Regression:

# remove the const column
X = X.drop('const', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# instantiate the regressor class
regressor = LinearRegression()

# build the model by fitting the regressor to the training data
regressor.fit(X_train, y_train)

# make a prediction set using the test set
prediction = regressor.predict(X_test)

# evaluate the prediction accuracy of the model
print("The Explained Variance: %.2f" % regressor.score(X_test, y_test))
print("The Mean Absolute Error: %.2f°C" % mean_absolute_error(y_test, prediction))
print("The Median Absolute Error: %.2f°C" % median_absolute_error(y_test, prediction))