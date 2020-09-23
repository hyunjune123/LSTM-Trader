import pandas as pd
import alpha_vantage
from alpha_vantage.timeseries import TimeSeries
import time
import qlearn
import random
from make_inputs import make_indicator_df


def main():
    print("Q-learning Day Trade Back Testing Program")
    print("Demo ver")
    api_key = '8TMSJT1WD6USMGI1'

    ts = TimeSeries(key=api_key, output_format='pandas')

    # Get data based on interval
    while(1):
        symb = str(input("Type stock symbol (Ex. for microsoft type MSFT) : "))
        #symb = "MSFT"
        if symb == "quit": exit(0)
        try:
            data, meta_data = ts.get_daily(symbol=symb, outputsize='full')
            break
        except:
            print("Invalid symbol")
    data.columns = ["Open", "High", "Low", "Close", "Volume"]
    data = data.iloc[::-1]
    print(data)
    train_len = int(input("Type number of days for training :"))
    test_len = int(input("Type number of dats for testing :"))
    principal = int(input("Type principal $ :"))
    window = int(input("Window :"))
    batch_len = int(input("Batch Length :"))
    #train_len = 600
    #test_len = 100
    #principal = 10000
    #window = 10
    #batch_len = 32

    # Data preparation for training / testing
    total_len = train_len + test_len
    train_start = random.randrange(0,len(data)-total_len)
    train_start = 0
    train_end = train_start + train_len
    test_end = train_end + test_len
    print("Data retrieved...")
    data = make_indicator_df((data[train_start:test_end]).reset_index()).dropna()
    indicators = data.columns[6:]
    print("Currently using indicators : ")
    print(indicators)

    trial_name = input("Test name : ")
    trial_num = input("Repeat time : ")
    # Q learning starts
    for i in range(0,int(trial_num)):
        qlearn.qlearn(data, window, principal, batch_len, train_len, indicators, str(i), trial_name)

if __name__ == "__main__":
    main()
