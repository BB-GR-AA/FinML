# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 10:11:17 2020
Temporary script for making input and output labels

@author: Gonzalo
"""

def halve_dataset(dataset): # Move to FinML
    ''' Returns the upper and lower halves of a data set.'''
    return dataset[:len(dataset)//2], dataset[len(dataset)//2:]

def make_input_output(dataset, q=21, r=1, s=1):
    return (inputs,outputs)

# Make data.
tmp =     
tmp2 = [tmp[i*q:(1+i)*q] for i in range(len(tmp))] # make sure they are tensors. try reshaping.
 