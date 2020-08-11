'''
Forecasting of stock market using linear regression implemented with Pytorch.
Visualization of the training process with TensorBoard.
'''
#stock = GetHistoricalData_AV(symbol='IBM')

# https://www.digitalocean.com/community/tutorials/how-to-import-modules-in-python-3#checking-for-and-installing-modules
# https://www.alphavantage.co/documentation/
# https://github.com/RomelTorres/alpha_vantage
# https://www.youtube.com/watch?v=TfuJSXTE9Rk
# https://github.com/cryptopotluck/alpha_vantage_tutorial/blob/master/_alpha_vantage/alpha_dataframe.py
# https://stackoverflow.com/questions/33157522/create-pandas-dataframe-from-dictionary-of-dictionaries

##import alpha_vantage
from alpha_vantage.timeseries import TimeSeries
import os.path
#import pandas as pd


### Load stock data  ### 

def GetHistoricalData_AV(symbol='IBM'):
##### Move GetHistoricalData to FinML.py get path working#####    
    keyfile = open(os.path.dirname(__file__) + '/../AVAPIK.txt','r')
    API_Key = keyfile.readline().rstrip('\n')
    ts = TimeSeries(key=API_Key,output_format='pandas')
    data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
    data.rename(columns = {'1. open':'open','2. high':'high','3. low':'low',
                           '4. close':'close','5. volume':'volume'})
    data.sort_values(by='date', ascending=True, inplace=True)
    return data


# keyfile = open(os.path.dirname(__file__) + '/../AVAPIK.txt','r')
# API_Key = keyfile.readline().rstrip('\n')
# ts = TimeSeries(key=API_Key,output_format='pandas')
# data, meta_data = ts.get_daily(symbol='IBM', outputsize='full')

data = GetHistoricalData_AV(symbol='IBM')
# df=pd.DataFrame([{'Label': 'Acura', 'Value': '1'}, {'Label': 'Agrale', 'Value': '2'}])
# df=df.rename(index=str, columns={"Label": "Make", "Value": "Code"})
# df.to_dict('records')
#data.rename(columns = {'1. open':'open','2. high':'high','3. low':'low',
#                       '4. close':'close','5. volume':'volume'})
#data.sort_values(by='date', ascending=True, inplace=True)



