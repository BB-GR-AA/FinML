'''
Training: google collab/cloud (IBM, google, AWS, Microsoft).
Visualization: TensorBoard.
Diagrams: inkscape.
Report: Jupyter Notebook (theory/method/citations and key findings).

Lunes:
- isort and Analyze results (train/test error) make sure they make sense (colud visualize predicted prices)
- Jupyter notebook (follow IBM notes and IBM notebooks markdown) save model and load.
'''

import sys
sys.path.append('../')
import torch.nn as nn
import torch.optim
from FinML import ANN
from FinML import DataSupervised
from torch.utils.data import DataLoader


### Training/Testing functions ### (move to FinML.py)   

def test_series(test_loader, model, criterion):
    """
        documentation
        Check how error is computed for batches and if we need to normalize.
    """        
    
    model.eval()
    error = 0   
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            error += criterion(model(x), y).item()  
            # Tensorboard and/or plot training
            #100 * (total / len(validation_loader))      
    model.train()
          
    return error 


def train_series(train_loader, model, criterion, optimizer, epochs=10, display_batch=False, test_loader=None):
    """
        Function for training a time series regression model.
        Make sure errors represent what you want.
        Move to FinML
        Training and testing errors should be on the same scale
        Write description.
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
            total += loss.item()  # cumulative loss    
            # Tensorboard and/or plot training
        results['training loss'].append(total) 
        
    if test_loader is not None:
        results['validation error'].append(test_series(test_loader, model, criterion))
        
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

model = ANN(Layers=[in_features, 10, 10, out_features])   
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

results = train_series(train_loader, model, criterion, optimizer, num_epochs, validation_loader)


### Analyze Results ###

