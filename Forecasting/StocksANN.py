'''
Training: google collab/cloud (IBM, google, AWS, Microsoft).
Visualization: TensorBoard.
Diagrams: inkscape.
Report: Jupyter Notebook (theory/method/citations and key findings).

Lunes:
- finish useful tutorials: https://www.youtube.com/playlist?list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz
- isort and Analyze results make sure training function is working properly.
- Jupyter notebook (follow IBM notes and IBM notebooks markdown) save model and load.

Next:
Tensorboard
    - https://www.tensorflow.org/tensorboard/get_started
    - https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
    - https://www.youtube.com/watch?v=VJW9wU-1n18&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=17&t=0s&ab_channel=PythonEngineer
- Change input size (cite stocks/timeseries study) Make wider and deeper (read and cite, ask stack exchange and ask literature) run and analyze.
- Cite transformations to standardize (try price difference P_i-p_{i-1} (input and max min scaler ooutput) univariate time series forecasting using neural networks, implement in dataset and run again.
- Compare tanh with default, tanh with xavier and relu with He (train all models, plot their loss and validations. (read and cite).
- Add drop out (read and cite).Dropout: a simple way to prevent neural networks from overfitting
- Compare batch normalization with dropout (5.3 lab).
- https://www.youtube.com/watch?v=P6NwZVl8ttc&t=96s&ab_channel=PyTorch
'''

import sys
sys.path.append('../')
import torch.nn as nn
import torch.optim
from FinML import ANN
from FinML import DataSupervised
from torch.utils.data import DataLoader


### Training function ### (move to FinML.py)   
def train(train_loader, validation_loader, model, criterion, optimizer, epochs=10, display_batch=False):
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


### Hyperparameters ###

learning_rate = 0.001
batch_size = 1
num_epochs = 2


### Custom Dataset and DataLoader ###   
 
train_dataset = DataSupervised(file_name="../Data/ANET_2014-06-06_2020-08-18_supervised_lag_21.csv")
validation_dataset = DataSupervised(file_name="../Data/ANET_2014-06-06_2020-08-18_supervised_lag_21.csv", train=False)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)

in_features = len(list(train_dataset.__getitem__(0)[0]))
out_features = len(list(train_dataset.__getitem__(0)[1]))


### Train and Test the Model ###

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ANN(Layers=[in_features, 10, 10, out_features]).to(device)    
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

results = train(train_loader, validation_loader, model,
                criterion, optimizer, epochs=num_epochs, display_batch=False)


### Analyze Results ###

