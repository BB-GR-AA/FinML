'''
Note: Inkscape is great for diagrams.
Stock (close) price forecast using a simple ANN implemented with Pytorch.
Training with google collab/cloud (IBM, google, AWS, Microsoft). Project next couple of months.
Visualization of the training process and architecture with TensorBoard.
Overview of Classes in Python.
Useful tutorials: https://www.youtube.com/playlist?list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz
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
from torch.utils.data import Dataset, DataLoader # For easier data set management (min batches, etc)
  


### Model architecture ###

# To do: Read on choice of number of layers and neurons. Simplest for now (one hidden layer).
# To do: Review classes in Python.

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
in_size = 21 # Number of days on which the prediction is based.
out_size = 1 # Number of days to be predicted.
learning_rate = 0.001 # step size
batch_size = 32 # https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network
num_epoch = 1 # 1 epoch means the network has seen the complete data set (Seen all batches).



### Load data, pre-processing and train/test sets ###

# To do: Start with halves then try alternating and random train test windows.
# To do: Save the test/train sets.

# Want batches because its not efficient to compute the gradient using the whole data set, for large sets.
# Training goes through each batch at once, compute that gradient and update weights.
# More formally:
#    - one epoch: One fwd and one bckwd pass of ALL the training examples.
#    - batch size: Number of training examples in one fwd/bckwd pass. Higher batch size, more memory.
#    - iterations: Number of passes each using [batch size] number of examples. One pass = one fwd + one bckwd.
#    ie. For 1,000 sample points and batch_size = 500, it will take two iterations to do one epoch.
    
def halve_dataset(dataset): # Move to FinML
    ''' Returns the upper and lower halves of a data set.'''
    return dataset[:len(dataset)//2], dataset[len(dataset)//2:]

def make_input_output(dataset, q=21, r=1, s=1): # Move to FinML
    ''' Sliding window size q, output size r and step size s.
    default values: q = 21, r = 3, s = 1.'''
    # To do: stress test this function. 
    # To do: Define constraints on q and r (ie size in relation to data set size).
    # assert len(dataset) > 
    #assert 1<r, "Window must contain at least two samples: (1<r)."
    # assert 0<s and s<=r, "Sliding parameter must be: (0<s<=r)."
    # this has to be compatible with trainloader. see if available time series data.
    # https://www.youtube.com/watch?v=mUueSPmcOBc&ab_channel=deeplizard
    # https://www.youtube.com/watch?v=PXOzkkB5eH0&ab_channel=PythonEngineer
    # https://www.youtube.com/watch?v=sCsPzVumtR8&ab_channel=PyTorch
    # https://pytorch.org/docs/stable/data.html
    # https://pytorch.org/docs/stable/data.html
    # https://www.kaggle.com/pinocookie/pytorch-dataset-and-dataloader
    return (inputs,outputs)

# Custom DataLoader

# Porbaly split, make input and output first and then make the CustomDataset object
class CustomDataset(Dataset): # Check DataSet class. Maybe also input training/testing sets. Move to FinML.
    def __init__(self):
        data = # Load data
        self.len = data.shape[0]
        self.data_x = # torch tensor
        self.y_data = # torch tensor
        file_name = '../Data/ANET_2014-06-06_2020-08-18.csv'
        stock = pd.read_csv(file_name, index_col=0) # use 'date' as index column.
        close = FinML.standardize(stock['close'])  # I dont think needed here since only one type of feature.
        close = torch.tensor(close) # create tensor from series. 
        
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index] 
    
    def __len__(self):
        return self.len 

# Ultimately want this (repeat for test half). Refactor where possible.   

# Split into train and test data.
train_dataset, _ = halve_dataset(close)

# Make input and output
train_dataset = make_input_output(train_dataset, q=21, r=1, s=1) # Get samples and labels for training.

# Object for DataLoader
train_dataset = customDataset(train_dataset)

# 
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=1) # Iterable.



### Train/test model ###

# To do: def train(Din,Dout,model,dataLoader,optimizer,loss,epochs) and add to FinML.py
# To do: Read Adam.

model = ANN(in_size,out_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

# Training cycle.
for epoch in range(num_epoch):
    # Loop over all batches in the training loader.
    for batch_idx, (data, targets) in enumerate(train_loader): # Get more familiar witht his line.
        inputs, labels = data # get the inputs
        inputs, labels = Variable(inputs), Variable(labels) # wrap them in Variable
        print(epoch, i, "inputs", inputs.data, "labels", labels.data) # Run training process.



# # save model and import it.
# # Plot of predicted and oserved data.
# # Next: LSTM, AR, Deep AR, esemble.

