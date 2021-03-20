from datetime import datetime, timedelta
import time
from collections import namedtuple
import pandas as pd
import requests
import matplotlib.pyplot as plt
import json

features = ["time_obs", "tmp_air_dry", "tmp_air_wet", "tmp_dew_pnt",
            "prs_lvl_hgt", "prs_sea_lvl", "wind_dir", "wind_spd"]

renamedFeatures = ["date", "meantempm", "meandewptm", "meanpressurem", "maxhumidity", "minhumidity",
                "maxtempm", "mintempm", "maxdewptm", "mindewptm", "maxpressurem", "minpressurem"]

DailySummary = namedtuple("DailySummary", renamedFeatures)

def sum_field(field, data):
    total = 0
    for d in data:
        total += d[field]
    return total

def calc_mean(field, data):
    return sum_field(field, data)/8

def calc_max(field, data):
    return max(x[field] for x in data)

def calc_min(field, data):
    return min(x[field] for x in data)

def calc_DailySummary(data):
    return DailySummary(
        date = data[0]['time_obs'],
        meantempm = calc_mean('tmp_air_dry', data),
        meandewptm = calc_mean('tmp_dew_pnt', data),
        meanpressurem = calc_mean('prs_lvl_hgt', data),
        maxhumidity = calc_max('hmd_rlt', data),
        minhumidity = calc_min('hmd_rlt', data),
        maxtempm = calc_max('tmp_air_dry', data),
        mintempm = calc_min('tmp_air_dry', data),
        maxdewptm = calc_max('tmp_dew_pnt', data),
        mindewptm = calc_min('tmp_dew_pnt', data),
        maxpressurem = calc_max('prs_lvl_hgt', data),
        minpressurem = calc_min('prs_lvl_hgt', data),
    )

def extract_weather_data(filename):
    records = []
    with open(filename) as json_file:
        data = json.load(json_file)
    for i in range(0, len(data), 8):
        currData = data[i:i+8]
        records.append(calc_DailySummary(currData))
    return records


# Getting the records from the json data file
records = extract_weather_data('data.json')
# print(records[0])

# Setting up Pandas DataFrame
df = pd.DataFrame(records, columns=renamedFeatures).set_index('date')

def derive_nth_day_feature(df, feature, N):
    rows = df.shape[0]
    nth_prior_measurements = [None]*N + [df[feature][i-N] for i in range(N, rows)]
    col_name = "{}_{}".format(feature, N)
    df[col_name] = nth_prior_measurements

for feature in renamedFeatures:
    if feature != 'date':
        for N in range(1, 4):
            derive_nth_day_feature(df, feature, N)

# Data cleaning
#
# make list of original features without meantempm, mintempm and maxtempm
features_to_remove = [feature for feature in renamedFeatures if feature not in ['meantempm', 'mintempm', 'maxtempm']]

# make a list of columns to keep
columns_to_keep = [col for col in df.columns if col not in features_to_remove]

# select only the columns in columns_to_keep and assign to df
df = df[columns_to_keep]
print(df.columns)

print(df.info())

# Call describe on df and transpose it due to the large number of columns
spread = df.describe().T

# precalculate interquartile range for ease of use in next calculation
IQR = spread['75%'] - spread['25%']

# create an outliers column which is either 3 IQRs below the first quartile or
# 3 IQRs above the third quartile
spread['outliers'] = (spread['min']<(spread['25%']-(3*IQR)))|(spread['max'] > (spread['75%']+3*IQR))

# just display the features containing extreme outliers
print(spread.loc[spread.outliers,])

plt.rcParams['figure.figsize'] = [14, 8]
df.maxhumidity_1.hist()
plt.title('Distribution of maxhumidity_1')
plt.xlabel('maxhumidity_1')
plt.show()

df.minpressurem_1.hist()
plt.title('Distribution of minpressurem_1')
plt.xlabel('minpressurem_1')
plt.show()