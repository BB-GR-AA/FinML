"""
Request historical data for a specific ticker and save to Data directory.
Update README.
"""

import os
import sys
sys.path.append('../')
import FinML

### Save data as csv ###

symbol = input("Enter ticker symbol: ").upper()
API_key = open(os.path.relpath("../AVAPIK.txt"), "r").readline().rstrip('\n')
stock = FinML.GetHistoricalData_AV(API_key, symbol=symbol)

start_date = str(stock.index.min().date())
end_date = str(stock.index.max().date())
file_name = symbol+'_'+start_date+'_'+end_date+'.csv'
stock.to_csv(file_name, index=True, index_label=stock.index.name)

print("Saved historical data for "+symbol+".")