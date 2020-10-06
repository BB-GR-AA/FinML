"""
Reframe daily stock close prices (time series) as a supervised learning datasets.
"""

import pandas as pd
import numpy as np

while True:
    file_name = input("Enter historical daily close price dataset from this directory: ")
    try:        
        stock = pd.read_csv(file_name, index_col=0) # use 'date' as index column.
        stock = stock['close'] # Keep only close prices.
    except ValueError:
        print("Sorry, I didn't understand that.")
        continue
    except OSError:
        print("No such file.")
        continue
    else:            
        try:
            lag = int(input("Enter number of past price samples (lag): "))                
        except ValueError:
            print("Enter a number.")
            continue
        if lag == "":
            continue              
        else:
            if lag < 2:
                print("The lag must contain at least two samples.")  
            else:
                break
              

d = np.zeros((stock.shape[0]-lag, lag+1)) 
for i in range(d.shape[0]):
    d[i,:] = stock[i:lag+i+1]
    
file_name = file_name[:-4]+'_supervised_lag_'+str(lag)+'.csv'        
np.savetxt(file_name, d, delimiter=",")
print("Saved historical daily price supervised dataset.")
