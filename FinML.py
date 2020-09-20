'''
Module with useful fucntions and classes.
'''

from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt 
import torch
from torch.utils.data import Dataset

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
  
class DataSupervised(Dataset):
    ''' Custom dataset to work with DataLoader for supervised learning.
    XY - The complete dataset as panadas dataframe.
    feature_cols - The column indices of the features as list, all but the last column as default.
    label_cols  - The column indices of the label as list, last column by default.'''
    
    def __init__(self, XY, feature_cols=None, label_cols=-1):
        if feature_cols != None :
            self.X = torch.tensor(XY.iloc[:,feature_cols].values) # check.view()
        else :
            self.X = torch.tensor(XY.iloc[:,:-1].values) # check.view()
        self.Y = torch.tensor(XY.iloc[:,label_cols]) # check.view()
        self.n_samples = self.X.shape[0]
        
    def __getitem__(self, index):
        return self.X[index], self.Y[index] 
    
    def __len__(self):
        return self.n_samples     
    
    
    