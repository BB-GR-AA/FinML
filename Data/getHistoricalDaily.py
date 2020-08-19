"""
Request historical daily price data for a specific ticker and to Data directory.
Update README.
"""

import os
import sys
sys.path.append('../')
import FinML


API_key = open(os.path.relpath("../AVAPIK.txt"), "r").readline().rstrip('\n')

while True:
    try:
        symbol = input("Enter ticker symbol: ").upper()
    except ValueError:
        print("Sorry, I didn't understand that.")
        continue
    if symbol == "":
        continue
    else:
        try:
            stock = FinML.GetHistoricalData_AV(API_key, symbol=symbol)
        except ValueError:
            print("Invalid API call. Please retry or visit the documentation:\n"
                  "(https://www.alphavantage.co/documentation/) for TIME_SERIES_DAILY")
            continue
        break    
    
start_date = str(stock.index.min().date())
end_date = str(stock.index.max().date())
file_name = symbol+'_'+start_date+'_'+end_date+'.csv'
stock.to_csv(file_name, index=True, index_label=stock.index.name)

print("Saved historical daily price data for "+symbol+".")




