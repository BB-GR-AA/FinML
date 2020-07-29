'''
Forecasting of stock market with linear regression implemented with Pytorch.
Visualization of the training process with TensorBoard.
'''
import numpy as np
import pandas as pd
import torch

# import pandas_datareader.data as web # Read what this import does. Find equivalent.
#https://pandas-datareader.readthedocs.io/en/latest/
#https://pandas-datareader.readthedocs.io/en/latest/remote_data.html
# import pandas_datareader.data
from datetime import datetime # import datetime.datetime ?
