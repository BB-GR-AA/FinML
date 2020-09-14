# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 10:11:17 2020
Temporary script for making input and output labels

@author: Gonzalo
"""

import torch


# def halve_dataset(dataset): # Move to FinML
#     ''' Returns the upper and lower halves of a data set.'''
#     return dataset[:len(dataset)//2], dataset[len(dataset)//2:]

# def make_input_output(dataset, window_size=21, prediction_size=1, slide_size=1):
#     return (inputs,outputs)

# Make data.
window_size=3
prediction_size=1
slide_size=1
D = torch.arange(22.) # Entire time series.
X = torch.reshape(D[:-1], (-1, window_size))
Y = D[torch.arange(window_size, D[-1]+window_size, window_size, dtype=int)]



tmp2 = [data[i*window_size:(1+i)*window_size] for i in range(len(data))] 


# Try with data not divisible by window size. 