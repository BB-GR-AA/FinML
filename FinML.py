""" Module with useful functions and classes."""


import matplotlib.pyplot as plt 
import pandas as pd
import torch
import torch.nn as nn
from alpha_vantage.timeseries import TimeSeries
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class ANN(nn.Module):
    """Artificial Neural Network
    
        MLP with tanh activation function for the hidden layers and linear transformation
        for the output layer by default.
        
        Layers -- (list) Numbers of neurons in each layer.
    """
    
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
        X_train, X_test = train_test_split(stock_supervised,test_size=0.2)         
        if train:
            self.X = torch.FloatTensor(X_train[:,:-target_cols])
            self.Y = torch.FloatTensor(X_train[:,-target_cols])
            if target_cols == 1:
                self.Y = self.Y.unsqueeze(1)            
        else:
            self.X = torch.FloatTensor(X_test[:,:-target_cols])     
            self.Y = torch.FloatTensor(X_test[:,-target_cols])
            if target_cols == 1:
                self.Y = self.Y.unsqueeze(1)                   
            
        self.n_samples = self.X.shape[0]
        
    def __getitem__(self, index):
        return self.X[index], self.Y[index] 
    
    def __len__(self):
        return self.n_samples     


def GetHistoricalData_AV(API_Key, symbol='IBM'):
    """Historical stock data as a pandas DataFrame."""
    
    ts = TimeSeries(key=API_Key,output_format='pandas')
    data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
    data.columns = ['open','high','low','close','volume']
    data.sort_values(by='date', ascending=True, inplace=True)
    return data
    
    
def plot_results_regression(results, units, test=True):
    """Plot of training and/or testing results.
    
    results -- A dictionary with keys: 'training loss', 'validation error'
    units -- Units to be displayed in the y-axis.
    test -- display validation results (default False).
    """

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(results['training loss'], color)
    ax1.set_xlabel('epoch', color='k')
    ax1.set_ylabel('training loss ('+units+')', color=color)
    ax1.tick_params(axis='y', color=color)

    if test:    
        ax2 = ax1.twinx()  
        color = 'tab:blue'
        ax2.plot(results['validation error'], color)
        ax2.set_xlabel('epoch', color='k')
        ax2.set_ylabel('validation error ('+units+')', color=color) 
        ax2.tick_params(axis='y', color=color)
        fig.tight_layout()
        

def test_series(test_loader, model, criterion):
    """ test a a time-series model, returns the error.
    
    test_loader -- DataLoader object with the test dataset.
    model -- Neural Network to be evaaluated.
    criterion -- Loss function.
    """        
    
    model.eval()
    error = 0   
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            error += criterion(model(x), y).item()   # cummulative error/batch
    model.train()

    return error / batch_idx # ~ error over all testing samples


def train_series(train_loader, model, criterion, optimizer, epochs=10, test_loader=None, display_batch=False):
    """Train and test (optional) a time-series model, returns the loss at a given epoch.

    train_loader -- DataLoader object with the training dataset.
    model -- Neural Network to be trained.
    criterion -- Loss function.
    optimizer -- optimization algorithm to update the network weights.
    epochs -- Number of forward and backward passes on the whole dataset.
    test_loader -- DataLoader object with the test dataset (default None).
    display_batch -- Display epoch, bactch index and batch length (default False).
    """
    
    model.train()
    results = {'training loss': [], 'validation error': []}   # loss at a given epoch        
    for epoch in range(epochs):
         
        total = 0 # training loss for every epoch        
        for batch_idx, (x, y) in enumerate(train_loader):             
            if display_batch: 
              print('epoch {}, batch idx {} , batch len {}'.format(epoch, batch_idx, len(y)))              
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item()  # cummulative loss/batch
        results['training loss'].append(total/batch_idx) # ~ loss over all training samples
        
        if test_loader is not None:
            results['validation error'].append(test_series(test_loader, model, criterion))
        
    return results


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
      
    
    
    
