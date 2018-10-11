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


def split_data_multimeter(df_in,drop_list, dummies, thresh = 1):
    ## Split and clean Data
    df_in.dropna(inplace = True)
    df = df_in.copy(deep = True)
    y = df['energy']
    X = df.drop(columns = drop_list, inplace = True)  ## removed time and tstp because i have a time_date object thats contains that data
    X = pd.get_dummies(df,columns = dummies)
    if thresh < 1:
        X = trimm_correlated(X,thresh)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    return X_train, X_test, y_train, y_test


def linear_reg_all(df, drop_list, dummies, thresh = 1):
    ## Split and clean Data
    X_train, X_test, y_train, y_test = split_data_multimeter(df,drop_list,dummies, thresh)

    # Fit your model using the training set
    linear = LinearRegression()
    lasso_cv = LassoCV(cv=5, random_state=0)
    ridge_cv = RidgeCV(alphas=(0.1, 1.0, 10.0))
    linear.fit(X_train, y_train)
    lasso_cv.fit(X_train, y_train)
    ridge_cv.fit(X_train, y_train)
    print('Linear regression score on train set with all parameters: {}'.format(linear.score(X_train, y_train)))
    print('Linear regression score on test set with all parameters: {}'.format(linear.score(X_test, y_test)))
    # print('Linear regression crossVal score on train set with all parameters: {}'.format(linear.score(X_train, y_train)))
    # print('Linear regression crossVal score on test set with all parameters: {}'.format(linear.score(X_test, y_test)))

    print('LassoCV regression score on train set with all parameters: {}'.format(lasso_cv.score(X_train, y_train)))
    print('LassoCV regression score on test set with all parameters: {}'.format(lasso_cv.score(X_test, y_test)))
    # print('LassoCV regression crossVal score on train set with all parameters: {}'.format(lasso_cv.score(X_train, y_train)))
    # print('LassoCV regression crossVal score on test set with all parameters: {}'.format(lasso_cv.score(X_test, y_test)))

    print('RidgeCV regression score on train set with all parameters: {}'.format(ridge_cv.score(X_train, y_train)))
    print('RidgeCV regression score on test set with all parameters: {}'.format(ridge_cv.score(X_test, y_test)))
    # print('RidgeCV regression crossVal score on train set with all parameters: {}'.format(ridge_cv.score(X_train, y_train)))
    # print('RidgeCV regression crossVal score on test set with all parameters: {}'.format(ridge_cv.score(X_test, y_test)))

    return ridge_cv, lasso_cv, linear, X_train, X_test, y_train, y_test



def trimm_correlated(df_in, threshold):
    df_corr = df_in.corr(method='pearson', min_periods=1)
    df_not_correlated = ~(df_corr.mask(np.tril(np.ones([len(df_corr)]*2, dtype=bool))).abs() > threshold).any()
    un_corr_idx = df_not_correlated.loc[df_not_correlated[df_not_correlated.index] == True].index
    df_out = df_in[un_corr_idx]
    return df_out


def plt_v_time(y_test, y_train, y_train_pred, y_test_pred):
    train_hr = np.arange(0,len(y_train))
    test_hr = np.arange(0,len(y_test))
    plt.figure(1)
    plt.scatter(train_hr,y_train, c ='r', label = 'Train Actual')
    plt.scatter(train_hr,y_train_pred, c= 'b', label = 'Train Predicted')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('LassoCV Predicted vs actual')
    plt.legend(loc='upper left')
    plt.figure(2)
    plt.scatter(test_hr ,y_test, c = 'r',label = 'Test Actual')
    plt.scatter(test_hr ,y_test_pred, label = 'Test Predicted')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('LassoCV Predicted vs actual')
    plt.legend(loc='upper left')
    plt.show()

def OLS_model(X_train,y_train):
    ols_model = sm.OLS(endog=y_train, exog=X_train).fit()
    print(ols_model.summary())
    energy_cons = ols_model.outlier_test()['energy']
    plt.figure(1)
    plt.scatter(ols_model.fittedvalues, energy_cons)
    plt.xlabel('Fitted values of AVGEXP')
    plt.ylabel('Studentized Residuals')
    plt.figure(2)
    sm.graphics.qqplot(energy_cons, line='45', fit=True)
    plt.show()


if __name__ == '__main__':
    dfmeter = pd.read_csv("data/smart_meters_london/daily_dataset/block_0.csv")
    dfweather = pd.read_csv("data/smart_meters_london/weather_daily_darksky.csv")

    dfmeter_hh = pd.read_csv("data/smart_meters_london/halfhourly_dataset/block_0.csv")##, dtype={'energy(kWh/hh)': float} )
    dfweather_h = pd.read_csv("data/smart_meters_london/weather_hourly_darksky.csv")


    df_meter_weather_hourly = join_df_weather(dfmeter_hh, dfweather_h) ## joins the two data frames on hour of time. Removes all non matching half hour data
    df_meter_weather_hourly['energy'] = pd.to_numeric(df_meter_weather_hourly['energy(kWh/hh)'])  ## converts exisitng energy colum to correct numeric values

##  No Time Values
    list_to_drop = ['energy','energy(kWh/hh)','time', 'tstp', 'date_start_time']
    Create_dummies = ['precipType', 'icon','summary', 'LCLid']
    ridge_cv, lasso_cv, linear, X_train, X_test, y_train, y_test = linear_reg_all(df_meter_weather_hourly, drop_list = list_to_drop, dummies = Create_dummies)

    print('Working on time Linear reg')

## With Time Values
    df_meter_weather_hourly['hour_column'] = [d.hour for d in df_meter_weather_hourly['date_start_time']]
    df_meter_weather_hourly['day_week'] = [d.dayofweek for d in df_meter_weather_hourly['date_start_time']]
    list_to_drop = ['energy','energy(kWh/hh)','time', 'tstp', 'date_start_time']
    Create_dummies = ['precipType', 'icon','summary', 'LCLid','day_week','hour_column']
    ridge_cv1, lasso_cv1, linear1, X_train1, X_test1, y_train1, y_test1 = linear_reg_all(df_meter_weather_hourly, drop_list = list_to_drop, dummies = Create_dummies)


    print('Working on time Linear reg with corr removal')

## With Time Values and removal of 0.7 corrolation
    list_to_drop = ['energy','energy(kWh/hh)','time', 'tstp', 'date_start_time']
    Create_dummies = ['precipType', 'icon','summary', 'LCLid','day_week','hour_column']
    threshold = 0.7
    ridge_cv2, lasso_cv2, linear2, X_train2, X_test2, y_train2, y_test2 = linear_reg_all(df_meter_weather_hourly, drop_list = list_to_drop, dummies = Create_dummies, thresh = threshold)

    plt_v_time(y_test2[0:100], y_train2[0:100], y_train_pred=lasso_cv2.predict(X_train2)[0:100] , y_test_pred=lasso_cv2.predict(X_test2)[0:100] )
