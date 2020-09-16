# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 10:11:17 2020
Temporary script for creating features and targets (labels).

@author: Gonzalo
"""

import torch


# Make data - Hankel matrix.(No need)
# Dont need Hankel, just need to create input matrix composed of window_size features by n_samples-w_s?
# and Y vector composed of n_samples-w_s by 1.
# For w_s consecutive prices (P_i, P_i+1, P_i+2,...,P_w_s) (these are the features
# predict P_w_s+1 (the label/tagert)
# After making data set of custom feautres and targets save the data set
# Split into training and testing set at random/alternating (halves)
# Read and cite how to split into training and testing data sets for time series.

window_size=3 # Features size.
prediction_size=1
slide_size=1
D = torch.arange(22.) # Entire time series.
X = torch.reshape(D[:-1], (-1, window_size))
Y = D[torch.arange(window_size, D[-1]+1, slide_size, dtype=int)]



# file_name = '../Data/ANET_2014-06-06_2020-08-18.csv'
# stock = pd.read_csv(file_name, index_col=0) # use 'date' as index column.
# close = FinML.standardize(stock['close'])  # I dont think needed here since only one type of feature.
# close = torch.tensor(close) # create tensor from series. 

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
#     # https://pytorch.org/docs/stable/data.html
#     # https://www.kaggle.com/pinocookie/pytorch-dataset-and-dataloader
#     return (inputs,outputs)
