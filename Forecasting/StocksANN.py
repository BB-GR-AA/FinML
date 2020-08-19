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

# # Split into train and test data.
# # Sliding window of size n and output of size k+1, start with k=0.
# # Class with simple ann architecture - torch.nn to define architecture and learn python classes.
# # Read a little on how to chose number of layers and neurons perlayer.
# # Train/test model - collab/Jax and tensorboard.
# # save model and import it.
# # Plot of predicted and oserved data.
# # Possible future ideas: CNN and Hankel matrix (FFT?) to predict k+1 outputs.
# # Possible future idea: more difficoult: candle stick chart image and output buy/sell

