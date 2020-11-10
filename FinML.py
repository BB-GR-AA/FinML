'''
Module with useful fucntions and classes.
'''

import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
from alpha_vantage.timeseries import TimeSeries
#from torch.utils.data import Dataset

class ANN(nn.Module):
    '''
        MLP with tanh activation function for the hidden layers and linear transformation
        for the output layer, boht by default.
        
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
      
    
    
    
