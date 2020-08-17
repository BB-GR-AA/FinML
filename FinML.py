'''
Module with useful tools.
'''
import os
from alpha_vantage.timeseries import TimeSeries

def GetHistoricalData_AV(API_Key, symbol='IBM'):
    '''Historical stock data as a pandas DataFrame. '''
    ts = TimeSeries(key=API_Key,output_format='pandas')
    data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
    data.columns = ['open','high','low','close','volume']
    data.sort_values(by='date', ascending=True, inplace=True)
    return data

def standardize(data):
    ''' Center data to the mean and element wise scale to unit variance.'''
    return (data - data.mean()) / data.std()
