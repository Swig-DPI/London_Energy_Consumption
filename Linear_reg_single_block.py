import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import plotly.plotly as py
import seaborn as sns

from sklearn.metrics import mean_squared_error #, cross_val_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV, Ridge
from fancyimpute import SimpleFill, KNN,  IterativeSVD, IterativeImputer
from plotly.graph_objs import *
from sklearn.preprocessing import scale
from statsmodels.stats.outliers_influence import variance_inflation_factor


def join_df_weather(dfmeter_hh, dfweather_h):
    '''dfmeter_hh =  half hour data from meters
    dfweather_h = hourly data from London
    Returns a merged list on date and time'''

    dfmeter_hh['date_start_time'] = pd.to_datetime(dfmeter_hh['tstp']) ## standardizes date and time for merge
    dfweather_h['date_start_time'] = pd.to_datetime(dfweather_h['time']) ## standardizes date and time for merge
    extradata = ['energy(kWh/hh)','time', 'tstp']

    ## Would be nice to return df with hour and half hour meter data with same weather database
        ## currently only returns hour incremented data
    merg = pd.merge(dfmeter_hh,dfweather_h, how = 'inner', left_on='date_start_time', right_on='date_start_time')
    # merg['energy'] = pd.to_numeric(merg['energy(kWh/hh)'])  ## converts exisitng energy colum to correct numeric values
    return merg # .drop(columns = extradata, inplace = True)


def split_data_multimeter(df_in):
    ## Split and clean Data
    df_in.dropna(inplace = True)
    df = df_in.copy(deep = True)
    y = df['energy']
    X = df.drop(columns = ['energy','energy(kWh/hh)','time', 'tstp', 'date_start_time'], inplace = True)  ## removed time and tstp because i have a time_date object thats contains that data
    X = pd.get_dummies(df,columns = ['precipType', 'icon','summary', 'LCLid'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    return X_train, X_test, y_train, y_test


def linear_reg_all(df):
    ## Split and clean Data
    X_train, X_test, y_train, y_test = split_data_multimeter(df)

    # Fit your model using the training set
    linear = LinearRegression()
    lasso_cv = LassoCV(cv=5, random_state=0)
    ridge_cv = RidgeCV(alphas=(0.1, 1.0, 10.0))
    linear.fit(X_train, y_train)
    lasso_cv.fit(X_train, y_train)
    ridge_cv.fit(X_train, y_train)
    print('Linear regression score on train set with all parameters: {}'.format(linear.score(X_train, y_train)))
    print('Linear regression score on test set with all parameters: {}'.format(linear.score(X_test, y_test)))
    print('Linear regression crossVal score on train set with all parameters: {}'.format(linear.score(X_train, y_train)))
    print('Linear regression crossVal score on test set with all parameters: {}'.format(linear.score(X_test, y_test)))

    print('LassoCV regression score on train set with all parameters: {}'.format(lasso_cv.score(X_train, y_train)))
    print('LassoCV regression score on test set with all parameters: {}'.format(lasso_cv.score(X_test, y_test)))
    print('LassoCV regression crossVal score on train set with all parameters: {}'.format(lasso_cv.score(X_train, y_train)))
    print('LassoCV regression crossVal score on test set with all parameters: {}'.format(lasso_cv.score(X_test, y_test)))

    print('RidgeCV regression score on train set with all parameters: {}'.format(ridge_cv.score(X_train, y_train)))
    print('RidgeCV regression score on test set with all parameters: {}'.format(ridge_cv.score(X_test, y_test)))
    print('RidgeCV regression crossVal score on train set with all parameters: {}'.format(ridge_cv.score(X_train, y_train)))
    print('RidgeCV regression crossVal score on test set with all parameters: {}'.format(ridge_cv.score(X_test, y_test)))

    return ridge_cv, lasso_cv, linear, X_train, X_test, y_train, y_test



if __name__ == '__main__':
    dfmeter = pd.read_csv("data/smart_meters_london/daily_dataset/block_0.csv")
    dfweather = pd.read_csv("data/smart_meters_london/weather_daily_darksky.csv")

    dfmeter_hh = pd.read_csv("data/smart_meters_london/halfhourly_dataset/block_0.csv")##, dtype={'energy(kWh/hh)': float} )
    dfweather_h = pd.read_csv("data/smart_meters_london/weather_hourly_darksky.csv")


    df_meter_weather_hourly = join_df_weather(dfmeter_hh, dfweather_h) ## joins the two data frames on hour of time. Removes all non matching half hour data
    df_meter_weather_hourly['energy'] = pd.to_numeric(df_meter_weather_hourly['energy(kWh/hh)'])  ## converts exisitng energy colum to correct numeric values

    ridge_cv, lasso_cv, linear, X_train, X_test, y_train, y_test = linear_reg_all(df_meter_weather_hourly)
