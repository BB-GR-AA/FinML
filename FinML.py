'''
Module with useful tools.
'''

from alpha_vantage.timeseries import TimeSeries

def GetHistoricalData_AV(symbol='IBM'):
    '''Historical stock data as a pandas DataFrame. '''
    API_Key = '8FF8W12C1V5Y69N9'
    ts = TimeSeries(key=API_Key,output_format='pandas')
    data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
    data.columns = ['open','high','low','close','volume']
    data.sort_values(by='date', ascending=True, inplace=True)
    return data
