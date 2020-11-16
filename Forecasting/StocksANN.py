'''
- Jupyter notebook (IBM notebooks). MSE formula for training/testing batches.
- Diagram of data splits. Add code snippets fomr FinML.
- candles plot of actual and predicted prices. save model and load.
- try largest dataset
- Why batches and how does loss and gradient get computed with batches? isort.

Noviembre:
- Cite transformations to standardize (try price difference P_i-p_{i-1} (input and max min scaler ooutput) univariate time series forecasting using neural networks, implement in dataset and run again.
- Change input size (cite stocks/timeseries study) Make wider and deeper (read and cite, ask stack exchange and ask literature) run and analyze.
- tanh with default vs tanh with xavier vs relu with He (loss and validations plots IBM notebooks, read and cite papers).
- drop out (IBM notebooks read and cite: Dropout: a simple way to prevent neural networks from overfitting)
- batch normalization vs dropout (5.3 lab).
'''

import sys
sys.path.append('../')
import torch.nn as nn
import torch.optim
import FinML
from FinML import ANN, DataSupervised
from torch.utils.data import DataLoader


### Hyperparameters ###

learning_rate = 0.001
batch_size = 32
num_epochs = 10


### Custom Dataset and DataLoader ###   
 
train_dataset = DataSupervised(file_name="../Data/ANET_2014-06-06_2020-08-18_supervised_lag_21.csv", train=True)
validation_dataset = DataSupervised(file_name="../Data/ANET_2014-06-06_2020-08-18_supervised_lag_21.csv", train=False)

train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size, shuffle=False)

in_features = len(list(train_dataset.__getitem__(0)[0]))
out_features = len(list(train_dataset.__getitem__(0)[1]))


### Train and Test the Model ###

model = ANN(Layers=[in_features, 10, 10, out_features])  
FinML.model_parameters_shape(model) 
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

results = FinML.train_series(train_loader, model, criterion, optimizer, num_epochs,
                       validation_loader, display_batch=False)


### Analyze Results ###

# Save image for notebook
FinML.plot_results_regression(results, units='$')
