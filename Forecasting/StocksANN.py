'''
Training: google collab/cloud (IBM, google, AWS, Microsoft).
Visualization: TensorBoard.
Diagrams: inkscape.
Report: Jupyter Notebook and Medium post (theory/method/citations and key findings).

To do:
- Run simplest architecture and Analyze results.
- Shuffle training for time series/stocks (read and cite, post stack exchange and ask literature).
- isort
- Read Adam optimizer SGD with/out momentum (cite) how does it work with batches?.
- finish useful tutorials: https://www.youtube.com/playlist?list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz
- Jupyter notebook (follow IBM notes and IBM notebooks markdown).
- Classes Python https://docs.python.org/3/tutorial/classes.html + https://www.youtube.com/playlist?list=PL-osiE80TeTsqhIuOqKhwlXsIBIdSeYtc
- Tensorboard
    - https://www.tensorflow.org/tensorboard/get_started
    - https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
    - https://www.youtube.com/watch?v=VJW9wU-1n18&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=17&t=0s&ab_channel=PythonEngineer
- Change input size (cite stocks/timeseries study) Make wider and deeper (read and cite, ask stack exchange and ask literature).
- Cite transformations to standardize (try price difference P_i-p_{i-1} (input and max min scaler ooutput) univariate time series forecasting using neural networks and implement in dataset.
- Non-linear autoencoder. start saving and importing models.
- Compare tanh with default, tanh with xavier and relu with He (train all models, plot their loss and validations. (read and cite).
- Add drop out (read and cite).Dropout: a simple way to prevent neural networks from overfitting
- Compare batch normalization with dropout (5.3 lab). Non-linear AE.
- https://www.youtube.com/watch?v=P6NwZVl8ttc&t=96s&ab_channel=PyTorch
- Next: vs LSTM (try ensemble) -> LSTM vs Deep AR -> LSTM/Deep AR vs ARIMA 
    - https://ieeexplore.ieee.org/document/8673351	
'''

import sys
sys.path.append('../')
import torch.nn as nn
import torch.optim
from FinML import ANN
from FinML import DataSupervised
from torch.utils.data import DataLoader


### Training function ### (move to FinML.py)   
def train(train_loader, validation_loader, model, criterion, optimizer, epochs=2, display_batch=False):
    """
        Make sure errors represent what you want.
        Move to FinML
        Training and testing errors should be on the same scale
        Write description.
    """
    results = {'training loss': [],'validation error': []}   # loss at a given epoch        
    for epoch in range(epochs):
         
        total = 0 # training loss for every epoch        
        for batch_idx, (x, y) in enumerate(train_loader):             
            if display_batch: 
              print('Training: epoch {}, batch idx {} , batch len {}'.format(epoch, batch_idx, len(y)))              
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item()  # cumulative loss    
            # Tensorboard and/or plot training
        results['training loss'].append(total) 
        
        total = 0 # validation loss for every epoch        
        for batch_idx, (x, y) in enumerate(validation_loader):
            if display_batch: 
              print('Validation: epoch {}, batch idx {} , batch len {}'.format(epoch, batch_idx, len(y)))            
            model.eval()
            total += criterion(model(x), y).item()  # cumulative loss 
            # Tensorboard and/or plot training
        #loss['validation error'].append((100 * (total / len(validation_loader)))          
        results['validation error'].append(total)          
          
    return results 


### Custom Dataset and DataLoader ###   
 
train_dataset = DataSupervised(file_name="../Data/ANET_2014-06-06_2020-08-18_supervised_lag_21.csv")
validation_dataset = DataSupervised(file_name="../Data/ANET_2014-06-06_2020-08-18_supervised_lag_21.csv", train=False)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=32, shuffle=False)


### Train and Test the Model ###

in_features = len(list(train_dataset.__getitem__(0)[0]))
model = ANN(Layers=[in_features, 10, 10, 1])    

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

results = train(train_loader, validation_loader, model,
                criterion, optimizer, epochs=100, display_batch=False)


### Analyze Results ###

