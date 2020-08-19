'''
Stock (close) price forecast using a simple ANN implemented with Pytorch.
Training with google collab/Jax.
Visualization of the training process and architecture with TensorBoard.
Overview of Classes in Python.
'''

import pandas as pd
import sys
sys.path.append('../')
import FinML
            
  
### Loading and pre-processing ###

file_name = '../Data/ANET_2014-06-06_2020-08-18.csv'
stock = pd.read_csv(file_name, index_col=0, usecols='close')
close = FinML.standardize(stock['close'])      

# # Sliding window of size n and output of size 1, the n+1-th sample.
# forecast_out = round(stock.shape[0] * 0.005)   # In days. Some % of the total data.
# forecast = stock['close'].shift(-forecast_out) # Predicted values.


# # Class with simple ann architecture - torch.nn to define architecture and learn python classes.
# # Train/test model - collab/Jax and tensorboard.
# # save model and import it.
# # Plot of predicted and oserved data.






