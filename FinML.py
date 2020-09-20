'''
Module with useful tools.
'''

from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt 

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

def PlotStuff(X,Y,model=None,leg=False):
    
    plt.plot(X[Y==0].numpy(),Y[Y==0].numpy(),'or',label='training points y=0 ' )
    plt.plot(X[Y==1].numpy(),Y[Y==1].numpy(),'ob',label='training points y=1 ' )

    if model!=None:
        plt.plot(X.numpy(),model(X).detach().numpy(),label='Neural Network ')

    plt.legend()
    plt.show()

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
    
    
    