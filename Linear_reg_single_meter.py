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



## Data Scrubber
def data_scrubber(income_df, percent_good):
    '''Simple data scrubber that fills the NAN columns with the mean
        of the data in the columns'''
    if(income_df.isna().sum().count() >= income_df.shape[0]*percent_good):
        print('Erroronious Data.  Look to fix')
        return(income_df.fillna(income_df.mean()))
    else:
        return(income_df.fillna(income_df.mean()))

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


def break_by_meter(df, unique_meters):
    '''df = weather and meter data combined
        Splits data by meter'''
    meter_df_list = []
    for meter_name in unique_meters:
        meter_df_list.append(df[df['LCLid'] == meter_name])

    return meter_df_list

## plot Scatter_matrix()
    ## Discuss corollations
def plot_scatter_matrix(df):
    plot1 = pd.plotting.scatter_matrix(df)
    plt.savefig('images/Scatter_matrix_of_{}.png'.format(df['LCLid'][0]))


## Mean energy by by block
    ## build function
    ## Plot heat map of blocks

## Mean energy by meter in each block
def mean_meter_all(meter_df_list):  ### Need to fixxx
    mean = []
    name = []
    for meter in meter_df_list:
        mean.append(meter['energy'].mean())
        name.append(meter['LCLid'].unique()[0]) ## this gets the unique name of the meter itself
    return mean, name



def split_data_single_meter(single_meter_df, drop_list):
    ## Split and clean Data
    single_meter_df.dropna(inplace = True)
    df1 = single_meter_df.copy(deep = True)
    y = df1['energy']
    X = df1.drop(columns = drop_list)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    return X_train, X_test, y_train, y_test, df1


def linear_reg_single_meter(X_train, X_test, y_train, y_test):
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

    return ridge_cv, lasso_cv, linear

def run_models(meter_df_list, meter,list_to_drop):
    X_train_meter, X_test_meter, y_train_meter, y_test_meter, df = split_data_single_meter(meter_df_list[meter_num],drop_list = list_to_drop)
    ridge_cv_meter, lasso_cv_meter, linear_meter  = linear_reg_single_meter(X_train_meter, X_test_meter, y_train_meter, y_test_meter)
    #OLS_model(X_train_meter,y_train_meter)
    print(OLS_model_noplot(X_train_meter,y_train_meter))
    return ridge_cv_meter, lasso_cv_meter, linear_meter, X_train_meter, X_test_meter, y_train_meter, y_test_meter


## This needs to be fixed.  Log issue??
def Ridge_plot(X_train, X_test, y_train, y_test):
    plot_y = []
    plot_y_test = []
    plot_x = np.arange(0.1,10,0.1)
    for val in np.arange(0.1,10,0.1):
        ridge = Ridge(alpha=val)
        ridge.fit(X_train.values, y_train.values)
        plot_y.append(ridge.score(X_train.values, y_train.values))
        plot_y_test.append(ridge.score(X_test.values, y_test.values))

    plt.scatter(plot_x, plot_y, c = 'r')
    plt.scatter(plot_x, plot_y_test, c = 'b')
    plt.show()
    return plot_x, plot_y, plot_y_test

## Not working on the data ?? need to fix
def OLS_model(X_train,y_train):
    ols_model = sm.OLS(endog=y_train, exog=X_train).fit()
    print(ols_model.summary())
    energy_cons = ols_model.outlier_test()['student_resid']
    plt.figure(1)
    plt.scatter(ols_model.fittedvalues, energy_cons)
    plt.xlabel('Fitted values of AVGEXP')
    plt.ylabel('Studentized Residuals')
    plt.figure(2)
    sm.graphics.qqplot(energy_cons, line='45', fit=True)
    plt.show()

def OLS_model_noplot(X_train,y_train):
    ols_model = sm.OLS(endog=y_train, exog=X_train).fit()
    print(ols_model.summary())


## Split all data into train and test.  Only  test on test
    ## Need function to read and write new files to split data
    ## Place in train and test file sets for easyc iterations
        ## should probably build a database (SQL) with tables by block number

## Add cost benifit matrix from confuion matrix see this: /Documents/galvanize/Lectures/lecture_profit-curve-imbal-classes

## For each block test data
    ## Set x and y values
    ## X is average meter data
    ## Y will be the weather and time (month and day) data

    ## split into train and test data

    ## Do LinearRegression with Kfolds
        ## QQ plot

    ## do logisticRegression with X above or below average use
        ## ROC plot

    #return linear errors

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
    plt.legend(loc='upper left')
    plt.show()



if __name__ == '__main__':
    dfmeter = pd.read_csv("data/smart_meters_london/daily_dataset/block_0.csv")
    dfweather = pd.read_csv("data/smart_meters_london/weather_daily_darksky.csv")

    dfmeter_hh = pd.read_csv("data/smart_meters_london/halfhourly_dataset/block_0.csv")##, dtype={'energy(kWh/hh)': float} )
    dfweather_h = pd.read_csv("data/smart_meters_london/weather_hourly_darksky.csv")


    df_meter_weather_hourly = join_df_weather(dfmeter_hh, dfweather_h) ## joins the two data frames on hour of time. Removes all non matching half hour data
    df_meter_weather_hourly['energy'] = pd.to_numeric(df_meter_weather_hourly['energy(kWh/hh)'])  ## converts exisitng energy colum to correct numeric values

    unique_meters = df_meter_weather_hourly['LCLid'].unique() ## gets unique meters in block
    meter_df_list = break_by_meter(df_meter_weather_hourly, unique_meters)

    meter_num = 0

    meter_df_list[meter_num] = pd.get_dummies(meter_df_list[meter_num],columns = ['precipType', 'icon','summary'])




## This is all for data with no Time parameter
    list_to_drop = ['energy','energy(kWh/hh)','time', 'tstp', 'date_start_time', 'LCLid','dewPoint', 'apparentTemperature']
    X_train_meter, X_test_meter, y_train_meter, y_test_meter, df = split_data_single_meter(meter_df_list[meter_num],drop_list = list_to_drop)
    ridge_cv_meter, lasso_cv_meter, linear_meter  = linear_reg_single_meter(X_train_meter, X_test_meter, y_train_meter, y_test_meter)
    #OLS_model(X_train_meter,y_train_meter)
    OLS_model_noplot(X_train_meter,y_train_meter)


    list_to_drop1 = list(X_test_meter.columns[lasso_cv_meter.coef_ == 0])
    list_to_drop = list_to_drop1 + list_to_drop
    X_train_meter1, X_test_meter1, y_train_meter1, y_test_meter1, df = split_data_single_meter(meter_df_list[meter_num], drop_list = list_to_drop )
    ridge_cv_meter1, lasso_cv_meter1, linear_meter1  = linear_reg_single_meter(X_train_meter1, X_test_meter1, y_train_meter1, y_test_meter1)
    #OLS_model(X_train_meter1,y_train_meter1)
    OLS_model_noplot(X_train_meter1,y_train_meter1)

    list_to_drop2 = list(X_test_meter1.columns[lasso_cv_meter1.coef_ == 0])
    list_to_drop = list_to_drop + list_to_drop2
    X_train_meter2, X_test_meter2, y_train_meter2, y_test_meter2, df = split_data_single_meter(meter_df_list[meter_num], drop_list = list_to_drop )
    ridge_cv_meter2, lasso_cv_meter2, linear_meter2  = linear_reg_single_meter(X_train_meter2, X_test_meter2, y_train_meter2, y_test_meter2)
    #OLS_model(X_train_meter2,y_train_meter2)
    OLS_model_noplot(X_train_meter2,y_train_meter2)

## Now add in the time parameter

    meter_df_list[meter_num]['hour_column'] = [d.hour for d in meter_df_list[meter_num]['date_start_time']]
    meter_df_list[meter_num]['day_week'] = [d.dayofweek for d in meter_df_list[meter_num]['date_start_time']]
    meter_df_list[meter_num] = pd.get_dummies(meter_df_list[meter_num] , columns = ['day_week','hour_column'])
    list_to_drop = ['energy','energy(kWh/hh)','time', 'tstp', 'LCLid','dewPoint', 'apparentTemperature', 'date_start_time']
    print(meter_df_list[meter_num].columns)
    X_train_meter3, X_test_meter3, y_train_meter3, y_test_meter3, df = split_data_single_meter(meter_df_list[meter_num], drop_list = list_to_drop)
    ridge_cv_meter3, lasso_cv_meter3, linear_meter3  = linear_reg_single_meter(X_train_meter3, X_test_meter3, y_train_meter3, y_test_meter3)
    #OLS_model(X_train_meter,y_train_meter)
    OLS_model_noplot(X_train_meter3,y_train_meter3)

    list_to_drop1 = list(X_test_meter.columns[lasso_cv_meter.coef_ == 0])
    list_to_drop = list_to_drop1 + list_to_drop
    X_train_meter4, X_test_meter4, y_train_meter4, y_test_meter4, df = split_data_single_meter(meter_df_list[meter_num], drop_list = list_to_drop )
    ridge_cv_meter4, lasso_cv_meter4, linear_meter4  = linear_reg_single_meter(X_train_meter4, X_test_meter4, y_train_meter4, y_test_meter4)
    #OLS_model(X_train_meter1,y_train_meter1)
    OLS_model_noplot(X_train_meter4,y_train_meter4)

    list_to_drop2 = list(X_test_meter1.columns[lasso_cv_meter1.coef_ == 0])
    list_to_drop = list_to_drop + list_to_drop2
    X_train_meter5, X_test_meter5, y_train_meter5, y_test_meter5, df = split_data_single_meter(meter_df_list[meter_num], drop_list = list_to_drop )
    ridge_cv_meter5, lasso_cv_meter5, linear_meter5  = linear_reg_single_meter(X_train_meter5, X_test_meter5, y_train_meter5, y_test_meter5)
    #OLS_model(X_train_meter2,y_train_meter2)
    OLS_model_noplot(X_train_meter5,y_train_meter5)

    print('use this')
    ridge_cv,lasso_cv,linear,X_train,X_test,y_train,y_test = run_models(meter_df_list, meter_num,list_to_drop)
    OLS_model_noplot(X_train_meter2,y_train_meter2)

    y_train_pred = lasso_cv.predict(X_train)
    y_test_pred = lasso_cv.predict(X_test)

    plt_v_time(y_test, y_train, y_train_pred, y_test_pred)

    #plot_x_alphas, plot_y, plot_y_test = ridgeCV_plot(X_train, X_test, y_train, y_test)
    mean_meter_enengy, mean_meter_enengy_name = mean_meter_all(meter_df_list)
    #heat_map_plot(mean_meter_enengy, mean_meter_enengy_name)
    #plt.show()
