import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from fancyimpute import SimpleFill, KNN,  IterativeSVD, IterativeImputer
import plotly.plotly as py
import plotly.graph_objs as go


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

        ## Would be nice to return df with hour and half hour meter data with same weather database
            ## currently only returns hour incremented data
        return pd.merge(dfmeter_hh,dfweather_h, how = 'inner', left_on='date_start_time', right_on='date_start_time')


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

def run_block(df):  ## this should be used to run all needed fucntions for each block
    pass


### Only use if you want to compute all blocks.
    ## Takes ~ 10 mins to run for 111 blocks
def means_of_all_blocks(number_of_blocks):
    '''This function will find the means of the n blocks in the data base'''
    dfweather_h = pd.read_csv("data/smart_meters_london/weather_hourly_darksky.csv")
    df_pass = pd.DataFrame({'A' : []})
    for i in range(number_of_blocks):
        print('block_{}'.format(i))
        string = "data/smart_meters_london/halfhourly_dataset/block_{}.csv".format(i)
        dfmeter_hh = pd.read_csv(string)
        df_meter_weather_hourly = join_df_weather(dfmeter_hh, dfweather_h) ## joins the two data frames on hour of time. Removes all non matching half hour data
        df_meter_weather_hourly['energy'] = pd.to_numeric(df_meter_weather_hourly['energy(kWh/hh)'])  ## converts exisitng energy colum to correct numeric values
        unique_meters = df_meter_weather_hourly['LCLid'].unique() ## Breaks into individual meter names
        meter_df_list = break_by_meter(df_meter_weather_hourly, unique_meters)  # See def
        mean_meter_enengy, mean_meter_enengy_name = mean_meter_all(meter_df_list) # see Def

        # This is to keep the mean lengths consistant
            ## should create a better way to implenment and add NAN instead of adding 0
                    # this will shift the mean down
        if len(mean_meter_enengy) == 50 :
            df_pass['block_{}'.format(i)] = mean_meter_enengy
        elif len(mean_meter_enengy) < 50:
            len_param = len(mean_meter_enengy)
            mean_meter_enengy = mean_meter_enengy + [0]*(50 -len_param)
            df_pass['block_{}'.format(i)] = mean_meter_enengy
        else:
            df_pass['block_{}'.format(i)] = mean_meter_enengy[0:50]

    df_pass.to_csv('data/smart_meters_london/meterAVGS.csv', encoding='utf-8', index=False)
    return df_pass.drop(labels = 'A', axis = 1)

def linear_reg(single_meter_df):
    y = single_meter_df['energy']
    X = single_meter_df.drop(labels = 'energy', axis = 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


    # Fit your model using the training set
    linear = LinearRegression()
    linear.fit(X_train, y_train)

    # Call predict to get the predicted values for training and test set
    train_predicted = linear.predict(X_train)
    test_predicted = linear.predict(X_test)



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
    # return Logisic errors

if __name__ == '__main__':
    dfmeter = pd.read_csv("data/smart_meters_london/daily_dataset/block_0.csv")
    dfweather = pd.read_csv("data/smart_meters_london/weather_daily_darksky.csv")

    dfmeter_hh = pd.read_csv("data/smart_meters_london/halfhourly_dataset/block_0.csv")##, dtype={'energy(kWh/hh)': float} )
    dfweather_h = pd.read_csv("data/smart_meters_london/weather_hourly_darksky.csv")


    df_meter_weather_hourly = join_df_weather(dfmeter_hh, dfweather_h) ## joins the two data frames on hour of time. Removes all non matching half hour data
    df_meter_weather_hourly['energy'] = pd.to_numeric(df_meter_weather_hourly['energy(kWh/hh)'])  ## converts exisitng energy colum to correct numeric values
    unique_meters = df_meter_weather_hourly['LCLid'].unique() ## gets unique meters in block
    meter_df_list = break_by_meter(df_meter_weather_hourly, unique_meters)

    plot_scatter_matrix(meter_df_list[0]) ## Initial scatter plot of a single meter at a single block
    linear_reg(df_meter_weather_hourly)
    mean_meter_enengy, mean_meter_enengy_name = mean_meter_all(meter_df_list)
    #heat_map_plot(mean_meter_enengy, mean_meter_enengy_name)
    #plt.show()
