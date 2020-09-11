'''
Note: Inkscape is great for diagrams.
Stock (close) price forecast using a simple ANN implemented with Pytorch.
Training with google collab/Jax/cloud.
Visualization of the training process and architecture with TensorBoard.
Overview of Classes in Python.
Tips: https://www.youtube.com/playlist?list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz
https://docs.python.org/3/tutorial/classes.html
'''

import pandas as pd
import numpy as np
import sys
sys.path.append('../')
import FinML
import torch # Entire library
import torch.nn as nn # All the neural network modules
import torch.optim # Optimization algorithms
from torch.utils.data import DataLoader # For easier data set management (min batches, etc)
  


### Model architecture ###

# To do: Read on choice of number of layers and neurons.
# To do: Overview of classes.

class ANN(nn.Module):
    # Inherit from nn.Module, the base class for all nn models.
    # All custom models will be a subclass of this class.
    def __init__(self, batch_size, forecast_size): # Define the initialization for our model.
        super(ANN,self).__init__() # Call initialization method of parent class. Super gives access to parent class.
        self.fc1 = nn.Linear(batch_size, 50) # Applies linear transofrmation, bias True by default.
        self.fc2 = nn.Linear(50, forecast_size) # Output layer.
    
    def forward(self,x): # x Is the data on which the linear transformation acts.
        x = nn.functional.relu(self.fc1(x)) # Linear and non-linear transformations in the first layer.
        x = self.fc2(x) # Linear transformations in the output layer.
        # No compeling reason to use activation function in output layer. Better to chose a suitable loss function.
        return x



### Device and hyperparameters ###

device = torch.device('cpu') # if gpu available do 'gpu'
in_size = 21 # Number of days on which the prediction is based.
out_size = 1 # Number of days to be predicted.
learning_rate = 0.001 # step size
batch_size = 32 # https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network



### Load data, pre-processing and train/test sets ###

# To do: Save the test/train sets.

file_name = '../Data/ANET_2014-06-06_2020-08-18.csv'
stock = pd.read_csv(file_name, index_col=0) # use 'date' as index column.
close = FinML.standardize(stock['close'])  # I dont think needed here since only one type of feature.
close = torch.tensor(close) # create tensor from series. 
    

def halve_dataset(dataset):
    # Move to FinML
    ''' Returns a the upper and lower halves of a data set.'''
    return dataset[:len(dataset)//2], dataset[len(dataset)//2:]

# Split into train and test data - Start with halves then try alternating train test windows, or random.
train_dataset, test_dataset = halve_dataset(close)

def make_input_output(dataset, q=21, r=1, s=1):
    ''' Sliding window size q, output size r and step size s.
    default values: q = 21, r = 3, s = 1.'''
    # Need to stress test this function in separate script for output based on inout parameters.
    # Need to test constraints on q and r, ie size in relation to data set size.
    # I think asserts need fixing.
    # assert len(dataset) > 
    #assert 1<r, "Window must contain at least two samples: (1<r)."
    # assert 0<s and s<=r, "Sliding parameter must be: (0<s<=r)."
    
    
    
tmp2 = [tmp[i*q:(1+i)*q] for i in range(len(tmp))] # make sure they are tensors. try rshaping.


# # Train/test model - collab/Jax/cloud and tensorboard.
# # save model and import it.
# # Plot of predicted and oserved data.
# # Next: CNN and Hankel matrix (FFT?) to predict k+1 outputs.
# # Next: LSTM

