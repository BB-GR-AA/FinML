# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 10:11:17 2020
Temporary script for making input and output labels

@author: Gonzalo
"""

import torch


# Make data - Hankel matrix.
window_size=3
prediction_size=1
slide_size=1
D = torch.arange(22.) # Entire time series.
X = torch.reshape(D[:-1], (-1, window_size))
Y = D[torch.arange(window_size, D[-1]+window_size, window_size, dtype=int)]



# Try with data not divisible by window size. 

# file_name = '../Data/ANET_2014-06-06_2020-08-18.csv'
# stock = pd.read_csv(file_name, index_col=0) # use 'date' as index column.
# close = FinML.standardize(stock['close'])  # I dont think needed here since only one type of feature.
# close = torch.tensor(close) # create tensor from series. 

# def halve_dataset(dataset): # Move to FinML
#     ''' Returns the upper and lower halves of a data set.'''
#     return dataset[:len(dataset)//2], dataset[len(dataset)//2:]

# def  make_input_output(dataset, window_size=21, prediction_size=1, slide_size=1):
#     # Move to FinML.
#     ''' Sliding window size q, output size r and step size s.
#     default values: window_size = 21, prediction_size = 1, slide_size = 1.'''
#     # Data size has to be divisible by window size, otherwise somehow deal with it.
#     # To do: stress test this function. 
#     # To do: Define constraints on parameters.
#     # assert len(dataset) > 
#     # assert 1<r, "Window must contain at least two samples: (1<r)."
#     # assert 0<s and s<=r, "Sliding parameter must be: (0<s<=r)."
#     # https://www.youtube.com/watch?v=mUueSPmcOBc&ab_channel=deeplizard
#     # https://www.youtube.com/watch?v=PXOzkkB5eH0&ab_channel=PythonEngineer
#     # https://pytorch.org/docs/stable/data.html
#     # https://www.kaggle.com/pinocookie/pytorch-dataset-and-dataloader
#     return (inputs,outputs)