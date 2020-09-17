'''
Stock daily close price prediction (Univariate one-step forecasting) using a simple ANN.

Training:  google collab/cloud (IBM, google, AWS, Microsoft).
Visualization: TensorBoard.
Diagrams: inkscape.
Summary: Jupyter Notebook (theory/method, citations and key findings)

To do:
- Overview of Classes in Python https://docs.python.org/3/tutorial/classes.html
- Read on choice of number of layers and neurons. Simplest for now (one hidden layer).
- Shuffle vs alternate for time series/stocks (read and cite)
- Read Adam optimizer
- Useful tutorials: https://www.youtube.com/playlist?list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz
'''

import pandas as pd
import numpy as np
import sys
sys.path.append('../')
import FinML
import torch # Entire library
import torch.nn as nn # All the neural network modules
import torch.optim # Optimization algorithms
from torch.utils.data import Dataset, DataLoader # For easier data set management (min batches, etc)
  


### Model architecture ###

class ANN(nn.Module):
    # Inherit from nn.Module, the base class for all nn models.
    # All custom models will be a subclass of this class.
    def __init__(self, input_size, forecast_size): # Define the initialization for our model.
        super(ANN,self).__init__() # Call initialization method of parent class. Super gives access to parent class.
        self.fc1 = nn.Linear(input_size, 50) # Applies linear transofrmation, bias True by default.
        self.fc2 = nn.Linear(50, forecast_size) # Output layer.
    
    def forward(self,x): # x Is the data on which the linear transformation acts.
        x = nn.functional.relu(self.fc1(x)) # Linear and non-linear transformations in the first layer.
        x = self.fc2(x) # Linear transformations in the output layer.
        # No compeling reason to use activation function in output layer. Better to chose a suitable loss function.
        return x



### Device and hyperparameters ###

device = torch.device('cpu') # Can change to gpu if available.
in_size = 21 # Number of days on which the prediction is based (features).
out_size = 1 # Number of days to be predicted.
learning_rate = 0.001 # step size
batch_size = 32 # https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network
num_epoch = 1 # 1 epoch means the network has seen the complete data set (Seen all batches).



### Load data, pre-processing and train/test sets ###

# Want batches because its not efficient to compute the gradient using the whole data set, for large sets.
# Training goes through each batch at once, compute that gradient and update weights.
# More formally:
#    - one epoch: One fwd and one bckwd pass of ALL the training examples.
#    - batch size: Number of training examples in one fwd/bckwd pass. Higher batch size, more memory.
#    - iterations: Number of passes each using [batch size] number of examples. One pass = one fwd + one bckwd.
#    ie. For 1,000 sample points and batch_size = 500, it will take two iterations to do one epoch.

# In this case we want something like this:
#   X is of size: number_samples-by-number_features
#   Y is of size: number_samples-by-1
#  [0,1,2]      [3]
#  [1,2,3]      [4]
#  [2,3,4]   =  [5]
#  [3,4,5]      [6]
#  [. . .]      [.]
#  [-4,-3,-2]   [-1]
    

# Custom Dataset class for DataLoader

class StockDataset(Dataset): # Training/testing sets as parameters. Move to FinML.
    def __init__(self):
        # data = # Load data
        self.n_samples = data.shape[0]
        self.x_data = # torch tensor
        self.y_data = # torch tensor
        
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index] 
    
    def __len__(self):
        return self.n_samples 

def halve_dataset(dataset): # Move to FinML
    ''' Returns the upper and lower halves of a data set.'''
    return dataset[:len(dataset)//2], dataset[len(dataset)//2:]

# Load custom dataset

# Shuffle dataset.

# Split into train and test data.
train_dataset, test_dataset = halve_dataset(data)

# Object for DataLoader
train_dataset = StockDataset(train_dataset)
test_dataset = StockDataset(test_dataset)
first_data_train = train_dataset[0]
features, targets = first_data_train
print(features, targets)

# DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=1) # Iterable.
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=1) # Iterable.

dataiter = iter(train_loader)
data = dataiter.next()
features, labels = data
print(features, targets)



### Train/test model ###

# To do: def train(Din,Dout,model,dataLoader,optimizer,loss,epochs) and add to FinML.py

# model = ANN(in_size,out_size).to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

# Training cycle.
# total_samples = lena(dataset)
# n_iter = math.ceil(total_samples/batch_size)
# print(total_samples, n_iter)

# for epoch in range(num_epoch):
#     # Loop over all batches in the training loader.
#     for batch_idx, (features, targets) in enumerate(train_loader): # Get more familiar witht his line.
#         inputs, labels = data # get the inputs
#         inputs, labels = Variable(inputs), Variable(labels) # wrap them in Variable
#         print(epoch, i, "inputs", inputs.data, "labels", labels.data) # Run training process.


# # save model and import it.
# # Plot of predicted and oserved data.
# # Next: LSTM, AR, Deep AR, ensemble.

