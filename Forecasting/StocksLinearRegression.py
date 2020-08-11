'''
Forecasting of stock market using linear regression implemented with Pytorch.
Visualization of the training process with TensorBoard.
'''

import os
import sys
sys.path.append('../')
import FinML

API_key = open(os.path.relpath("../AVAPIK.txt"), "r").readline().rstrip('\n')
stock = FinML.GetHistoricalData_AV(API_key, symbol='IBM')
