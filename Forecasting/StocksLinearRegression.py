'''
Forecasting of stock market using linear regression implemented with Pytorch.
Visualization of the training process with TensorBoard.
'''

import sys
sys.path.append('../')
import FinML

stock = FinML.GetHistoricalData_AV(symbol='IBM')


