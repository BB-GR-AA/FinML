'''
Forecasting of stock market using linear regression implemented with Pytorch.
Visualization of the training process with TensorBoard.
'''
import numpy as np
import os
import sys
sys.path.append('../')
import FinML


# API_key = open(os.path.relpath("../AVAPIK.txt"), "r").readline().rstrip('\n')
# stock = FinML.GetHistoricalData_AV(API_key, symbol='IBM')


forecast_out = round(stock.shape[0] * 0.005)          # In days. Some % of the total data.
forecast = stock['close'].shift(-forecast_out).values # Convert to np array.
close = stock['close'].values                         # Convert to np array.
close = FinML.standardize(close)                      # Center to the mean and scale to unit variance.
