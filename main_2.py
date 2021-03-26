from datetime import datetime, timedelta
import time
from collections import namedtuple
import pandas as pd
import requests
import matplotlib.pyplot as plt
import json
import numpy as np
import constants

features = ["time_obs", "tmp_air_dry", "tmp_air_wet", "tmp_dew_pnt",
            "prs_lvl_hgt", "prs_sea_lvl", "wind_dir", "wind_spd"]

renamedFeatures = ["date", "meantempm", "meandewptm", "meanpressurem", "maxhumidity", "minhumidity",
                "maxtempm", "mintempm", "maxdewptm", "mindewptm", "maxpressurem", "minpressurem"]

DailySummary = namedtuple("DailySummary", renamedFeatures)

def sum_field(field, data):
    total = 0
    count = 0
    for d in data:
        if d[field] != -9999:
            total += d[field]
            count += 1
        # total += d[field] if d[field] != -9999 else 0
        # total += d[field]
    return total, count
    # return total

def calc_mean(field, data):
    total, count = sum_field(field, data)
    return total/count if count!=0 else 0
    # if count != 0:
    #     return total/count
    # else:
    #     return 0
    # return sum_field(field, data)/8

def calc_max(field, data):
    return max(x[field] if x[field]!=-9999 else 0 for x in data)
    # return max(x[field] for x in data)

def calc_min(field, data):
    return min(x[field] if x[field]!=-9999 else 0 for x in data)

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
    # print(records[0])
    return records

def derive_goal_temperature_N_days_ahead(df, N):
    rows = df.shape[0]
    day_N_temperatures = [df['meantempm'][i+N] for i in range(0, rows-N)] + [None]*N
    col_name = col_name = 'day_{}_meantempm'.format(N)
    df[col_name] = day_N_temperatures

def derive_nth_day_feature(df, feature, N):
    rows = df.shape[0]
    nth_prior_measurements = [None]*N + [df[feature][i-N] for i in range(N, rows)]
    col_name = "{}_{}".format(feature, N)
    df[col_name] = nth_prior_measurements

def get_df():
    # Getting records from the json data file
    records = extract_weather_data('data.json')

    # Setting up Pandas DataFrame
    df = pd.DataFrame(records, columns=renamedFeatures).set_index('date')

    for feature in renamedFeatures:
        if feature != 'date':
            for N in range(1, 4):
                derive_nth_day_feature(df, feature, N)
    
    derive_goal_temperature_N_days_ahead(df, constants.N_DAY)
    
    # Make list of original features without meantempm, mintempm and maxtempm
    # features_to_remove = [feature for feature in renamedFeatures if feature not in ['meantempm', 'mintempm', 'maxtempm']]
    features_to_remove = [feature for feature in renamedFeatures if feature not in []]

    # # Make a list of columns to keep
    # columns_to_keep = [col for col in df.columns if col not in features_to_remove]

    # # Select only the columns in columns_to_keep and assign to df
    # df = df[columns_to_keep]
    # print(df.columns)

    # print(df.info())

    # Call describe on df and transpose it due to the large number of columns
    spread = df.describe().T

    # Precalculate interquartile range for ease of use in next calculation
    IQR = spread['75%'] - spread['25%']

    # Create an outliers column which is either 3 IQRs below the first quartile or
    # 3 IQRs above the third quartile
    spread['outliers'] = (spread['min']<(spread['25%']-(3*IQR)))|(spread['max'] > (spread['75%']+3*IQR))

    # Display the features containing extreme outliers
    # print(spread.loc[spread.outliers,])

    # plt.rcParams['figure.figsize'] = [14, 8]
    # df.maxhumidity_1.hist()
    # plt.title('Distribution of maxhumidity_1')
    # plt.xlabel('maxhumidity_1')
    # plt.show()

    # df.minpressurem_1.hist()
    # plt.title('Distribution of minpressurem_1')
    # plt.xlabel('minpressurem_1')
    # plt.show()

    # Drop records containing NaN values
    df = df.dropna()
    return df