'''
Module with useful functions and classes.
'''

import matplotlib.pyplot as plt 
import pandas as pd
import torch
import torch.nn as nn
from alpha_vantage.timeseries import TimeSeries
from torch.utils.data import Dataset


class ANN(nn.Module):
    '''
        MLP with tanh activation function for the hidden layers and linear transformation
        for the output layer by default.
        
        Layers (list): Numbers of neurons in each layer.
    ''' 
    
    def __init__(self, Layers):
        super(ANN, self).__init__()  
        self.hidden = nn.ModuleList()
        
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
    
    
class DataSupervised(Dataset):
    '''
        Custom dataset to work with DataLoader for supervised learning.
        
        file_name (str): Path to csv file.
        target_cols (int): Number of steps (forecasts), one by default (last column).
        train (bool): Train (odd samples) or test split (even samples), True by default.
    '''    
    
    def __init__(self, file_name, target_cols=1, train=True):
        
        stock_supervised = pd.read_csv(file_name).values
                
        if train:
            self.X = torch.FloatTensor(stock_supervised[::2,:-target_cols])
            self.Y = torch.FloatTensor(stock_supervised[::2,-target_cols])
            if target_cols == 1:
                self.Y = self.Y.unsqueeze(1)            
        else:
            self.X = torch.FloatTensor(stock_supervised[1::2,:-target_cols])     
            self.Y = torch.FloatTensor(stock_supervised[1::2,-target_cols])
            if target_cols == 1:
                self.Y = self.Y.unsqueeze(1)                   
            
        self.n_samples = self.X.shape[0]
        
    def __getitem__(self, index):
        return self.X[index], self.Y[index] 
    
    def __len__(self):
        return self.n_samples     


def GetHistoricalData_AV(API_Key, symbol='IBM'):
    '''Historical stock data as a pandas DataFrame. '''
    ts = TimeSeries(key=API_Key,output_format='pandas')
    data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
    data.columns = ['open','high','low','close','volume']
    data.sort_values(by='date', ascending=True, inplace=True)
    return data


def standardize(data):
    ''' Center data to the mean and element wise scale to unit variance.'''
    return (data - data.mean()) / data.std()


def plot_trainning(X, Y, model, epoch, leg=True, x_label='x', y_label='y'):
    ''' This function produces a plot of the predicted values and the training dataset.'''
    plt.plot(X.numpy(), model(X).detach().numpy(), label=('epoch ' + str(epoch)))
    plt.plot(X.numpy(), Y.numpy(), 'r', label='training dataset')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if leg == True:
        plt.legend()
    else:
        pass
    plt.show()


def print_model_parameters(model):
    count = 0
    for ele in model.state_dict():
        count += 1
        if count % 2 != 0:
            print ("The following are the parameters for the layer ", count // 2 + 1)
        if ele.find("bias") != -1:
            print("The size of bias: ", model.state_dict()[ele].size())
        else:
            print("The size of weights: ", model.state_dict()[ele].size())    
      
    
    
    
