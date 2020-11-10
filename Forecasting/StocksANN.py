'''
Training: google collab/cloud (IBM, google, AWS, Microsoft).
Visualization: TensorBoard.
Diagrams: inkscape.
Report: Jupyter Notebook and Medium post (theory/method/citations and key findings).

To do:
- Training function
- Shuffle training for time series/stocks (read and cite, post stack exchange and ask literature).
- isort
- Run simplest architecture and Analyze results.
- Read Adam optimizer SGD with momentum (cite).
- finish useful tutorials: https://www.youtube.com/playlist?list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz
- Jupyter notebook (follow IBM notes and IBM notebooks markdown).
- Classes Python https://docs.python.org/3/tutorial/classes.html + https://www.youtube.com/playlist?list=PL-osiE80TeTsqhIuOqKhwlXsIBIdSeYtc
- Tensorboard
    - https://www.tensorflow.org/tensorboard/get_started
    - https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
    - https://www.youtube.com/watch?v=VJW9wU-1n18&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=17&t=0s&ab_channel=PythonEngineer
- Change input size (cite stocks/timeseries study) Make wider and deeper (read and cite, ask stack exchange and ask literature).
- Cite transformations to standardize (try price difference P_i-p_{i-1} (input and max min scaler ooutput) univariate time series forecasting using neural networks and implement in dataset.
- Non-linear autoencoder.
- Compare tanh with default, tanh with xavier and relu with He (train all models, plot their loss and validations. (read and cite).
- Add drop out (read and cite).Dropout: a simple way to prevent neural networks from overfitting
- Compare batch normalization with dropout (5.3 lab). Non-linear AE.
- https://www.youtube.com/watch?v=P6NwZVl8ttc&t=96s&ab_channel=PyTorch
- Next: vs LSTM (try ensemble) -> LSTM vs Deep AR -> LSTM/Deep AR vs ARIMA 
    - https://ieeexplore.ieee.org/document/8673351	
'''

from FinML import ANN
from FinML import DataSupervised
from torch.utils.data import DataLoader


### Custom Dataset and DataLoader ###   
 
train_dataset = DataSupervised(file_name="../Data/ANET_2014-06-06_2020-08-18_supervised_lag_21.csv")
test_dataset = DataSupervised(file_name="../Data/ANET_2014-06-06_2020-08-18_supervised_lag_21.csv", train=False)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)


### Model ###

model = ANN(Layers=[train_dataset.n_samples, 10, 10, 1])


### Training function ### (move to FinML.py)

def train(Y, X, model, criterion, optimizer, , epochs=20):
    
    error = [] # loss at a given epoch    
    
    for epoch in range(epochs):        
        total = 0 # loss for entire data set for one epoch.
        
        # Do one full fwd and bckwd pass on entire data set.
        for y, x in zip(Y, X):
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += loss.item()  #cumulative loss 
            
        error.append(total)  
        
    return error
       
def train(model, criterion, train_loader, test_loader, optimizer, epochs, display_batch=false):
    
    i = 0
    loss = {'training_loss': [],'validation MSE': []}  
    
    for epoch in range(epochs):
        
        for batch_idx, (x, y) in enumerate(train_loader):             
            if display_batch : 
              print('epoch {}, batch_idx {} , batch len {}'.format(epoch, batch_idx, len(y)))    
              
            model.train()
            optimizer.zero_grad()
            z = model(x.view(-1, 28 * 28))
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
              #loss for every iteration
            error['training_loss'].append(loss.data.item())
        correct = 0
        
        # Tensorboard + plot training
        for i, (x, y) in enumerate(validation_loader):
            model.eval()
            z = model(x)
	      error = criterion(z, y)
            print(i)
            print('Predicted: ',z.item())
	      print('Actual: ', y.item())
   	      print('Loss: ', loss.item())
            loss['MSE?/RMSE? read which'].append(error)
          # Tensorboard + plot training
    return loss 


### Train and Test the Model ###