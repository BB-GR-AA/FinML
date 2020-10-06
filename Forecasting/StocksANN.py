'''
Training:  google collab/cloud (IBM, google, AWS, Microsoft).
Visualization: TensorBoard.
Diagrams: inkscape.
Report: Jupyter Notebook (theory/method, citations and key findings).
Summary: Medium - Towards Data Science.

To do:
- supervised learning dataset (Get into single csv)
- Dataset class
- Training function
- data loader https://medium.com/noumena/how-does-dataloader-work-in-pytorch-8c363a8ee6c1 move to FinML
- Shuffle training for time series/stocks (read and cite).
- Run simplest architecture.
- Analyze results.
- Jupyter notebook (follow IBM notes).
- Overview of Classes in Python https://docs.python.org/3/tutorial/classes.html + YT video.
- Read Adam optimizer SGD with momentum (cite).
- finish useful tutorials: https://www.youtube.com/playlist?list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz
- Tensorboard
    - https://www.tensorflow.org/tensorboard/get_started
    - https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
    - https://www.youtube.com/watch?v=VJW9wU-1n18&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=17&t=0s&ab_channel=PythonEngineer
- Make wider and deeper (read and cite).
- Compare tanh with default vs tanh with xavier vs relu with He (train all models, plot their loss and validations. (read and cite).
- Add drop out (read and cite).Dropout: a simple way to prevent neural networks from overfitting
- Batch normalization. Compare with dropout.
- Next: vs ARIMA -> LSTM vs ARIMA -> Deep AR vs ARIMA -> Ensemble. 
    - https://ieeexplore.ieee.org/document/8673351	
'''

import torch # Package with data structures for tensors and their mathematical operations.
import torch.nn as nn # Package for building and training neural networks.
import torch.optim # Optimization algorithms
import torchsummary
from torch.utils.data import Dataset, DataLoader # For easier data set management.
import pandas as pd
import numpy as np
import sys
sys.path.append('../')
import FinML

  
### Model architecture ###

class ANN(nn.Module):
    # Compare tanh and relu activation functions (read, cite) 
    # Compare PyTorch default initialization vs Xavier (read, cite)
    # Try He initialization vs defaul for ReLu (read, cite)
    def __init__(self, Layers):
        super(ANN, self).__init__()        
        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            self.hidden.append(linear)
    
    def forward(self, x):
        layers = len(self.hidden)
        for (layer, linear_transform) in zip(range(layers), self.hidden):
            if layer < layers - 1:
                x = torch.tanh(linear_transform(x))
            else:
                x = linear_transform(x)
        return x


### Training function ### (move to FinML.py)

# zip() creaters an iterator of tuples,  the i-th tuple being (x_i, y_i).
# enumerate() creates an iterator of tuples,  the i-th tuple being (counter_i, item_i). 

# def train(Y, X, model, optimizer, criterion, epochs=1000):
#     error = [] # loss at a given epoch
#     for epoch in range(epochs):
#         total=0 # loss for entire data set for one epoch.
#         # Do one full fwd and bckwd pass on entire data set.
#         for y, x in zip(Y, X):
#             loss = criterion(model(x), y)
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#             #cumulative loss 
#             total+=loss.item() 
#         error.append(total)         
#     return error
       
# def train(model, criterion, train_loader, validation_loader, optimizer, epochs=5, display_batch=false):
#     i = 0
#     useful_stuff = {'training_loss': [],'validation MSE?': []}  
#     for epoch in range(epochs):
#         for batch_idx, (x, y) in enumerate(train_loader): 
#             if display_batch : 
#               print('epoch {}, batch_idx {} , batch len {}'.format(epoch, batch_idx, len(y)))    
#             optimizer.zero_grad()
#             z = model(x.view(-1, 28 * 28))
#             loss = criterion(z, y)
#             loss.backward()
#             optimizer.step()
#              #loss for every iteration
#             useful_stuff['training_loss'].append(loss.data.item())
#         correct = 0
#         # Tensorboard + plot training
#         for x, y in validation_loader:
#             #validation 
#             z = model(x.view(-1, 28 * 28))
#             _, label = torch.max(z, 1)
#             correct += (label == y).sum().item()
#         accuracy = 100 * (correct / len(validation_dataset))
#         useful_stuff['MSE?'].append(accuracy)
#         # Tensorboard + plot training
#     return useful_stuff 


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

# All custom data sets that are fed into the class DataLoader must:
# interith the Dataset class ... but why?    
# constructor __init__()
        #if train != True:
            #return even rows
#        else:
            # return odd rows 
#         self.y is the last column
# __getitem__()
# __len__() 

#def halve_dataset(dataset): # Move to FinML
#    ''' Returns the upper and lower halves of a data set.'''
#    return dataset[:len(dataset)//2], dataset[len(dataset)//2:]

# Load custom dataset

# Split into train and test data.
# train_dataset, test_dataset = halve_dataset(data)

# Object for DataLoader
train_dataset = StockDataset(train_dataset)
test_dataset = StockDataset(test_dataset)
first_data_train = train_dataset[0]
features, targets = first_data_train
print(features, targets)

# DataLoader

# What exactly does DataLoader return?
# I think it returns an iterator of tuples (x,y) in batches.
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1) # Iterable.
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=1) # Iterable.

dataiter = iter(train_loader)
data = dataiter.next()
features, labels = data
print(features, targets)
# See if we can print bacth_size, iterations (number of batches).

# total_samples = lena(dataset)
# n_iter = math.ceil(total_samples/batch_size)
# print(total_samples, n_iter)

### Trainining/testing ###

# device = torch.device('cpu') # Can change to gpu if available.
# in_size = 7 # Number of days on which the prediction is based (features).
# H = 10 size of hidden layer
# out_size = 1 # Number of days to be predicted.
# learning_rate = 0.001 # step size
# batch_size = 32 # https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network
# num_epoch = 1 # 1 epoch means the network has seen the complete data set (Seen all batches).

# model = ANN(in_size,out_size).to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate) 


# # save model and import it.
# # Plot of predicted and oserved data.


