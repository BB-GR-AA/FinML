# Useful code for future sections

from torch.utils.data import Dataset, DataLoader # For easier data set management.
import pandas as pd
import numpy as np
import torch.optim # Optimization algorithms
import torchsummary
import FinML

### Model ###

#model = ANN(Layers=[2,2,1])

#def halve_dataset(dataset): # Move to FinML
#    ''' Returns the upper and lower halves of a data set.'''
#    return dataset[:len(dataset)//2], dataset[len(dataset)//2:]

# Load custom dataset

# Split into train and test data.
# train_dataset, test_dataset = halve_dataset(data)

# Object for DataLoader
# train_dataset = StockDataset(train_dataset)
# test_dataset = StockDataset(test_dataset)
# first_data_train = train_dataset[0]
# features, targets = first_data_train
# print(features, targets)

# DataLoader

# What exactly does DataLoader return?
# I think it returns an iterator of tuples (x,y) in batches.
# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1) # Iterable.
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=1) # Iterable.

# dataiter = iter(train_loader)
# data = dataiter.next()
# features, labels = data
# print(features, targets)
# See if we can print bacth_size, iterations (number of batches).

# total_samples = lena(dataset)
# n_iter = math.ceil(total_samples/batch_size)
# print(total_samples, n_iter)

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
#     loss = {'training_loss': [],'validation MSE?': []}  
#     for epoch in range(epochs):
#         for batch_idx, (x, y) in enumerate(train_loader): 
#             if display_batch : 
#               print('epoch {}, batch_idx {} , batch len {}'.format(epoch, batch_idx, len(y)))    
#             model.train()
#             optimizer.zero_grad()
#             z = model(x.view(-1, 28 * 28))
#             loss = criterion(z, y)
#             loss.backward()
#             optimizer.step()
#              #loss for every iteration
#             error['training_loss'].append(loss.data.item())
#         correct = 0
#         # Tensorboard + plot training
#         for i, (x, y) in enumerate(validation_loader):
#             model.eval()
#             z = model(x)
#	      error = criterion(z, y)
#             print(i)
#             print('Predicted: ',z.item())
#	      print('Actual: ', y.item())
#    	      print('Loss: ', loss.item())
#             loss['MSE?/RMSE? read which'].append(error)
#           # Tensorboard + plot training
#     return loss 

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
