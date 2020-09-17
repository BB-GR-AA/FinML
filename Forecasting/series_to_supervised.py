# -*- coding: utf-8 -*-
"""
Script for reframing time series datasets as a supervised learning problem.

Given a time series s(t), create a data set composed of inputs/outputs for supervised learning.
This data set d = {(x_i, y_i)}, where i=0,1,2,...,N will be used to build the model:
        s_hat(t) = f(s(t-1), s(t-2), ..., s(t-l))), such that s_hat(t) ~ s(t).


Terminology:
Raw data - s(t)
data set for deep learning - d(t)
Feature vector - x_i = s(t-1), s(t-2), ..., s(t-l)
Target - y_i
Lag - l
Forecast - s_hat(t)

Implementation:
- Explicitly build Hankel matrix (n_rows by l+1) from s(t)
- Split into X, Y
- Save the data set with appropriate name
- Move function to FinML.py
"""

import torch


window_size=3 # Features size.
prediction_size=1
slide_size=1
D = torch.arange(22.) # Entire time series.
X = torch.reshape(D[:-1], (-1, window_size))
Y = D[torch.arange(window_size, D[-1]+1, slide_size, dtype=int)]
https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/


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
