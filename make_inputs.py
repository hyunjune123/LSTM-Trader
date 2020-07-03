import numpy as np
import pandas as pd

# Helper functions
def mvg(prices, periods=20):
    weights = np.ones(periods) / periods
    return np.convolve(prices, weights, mode='valid')

def sd(prices, periods = 20):
    for i in range(0,len(prices)):
        if i < periods:
            return 0
    return 0

# df = df_mem
def make_input_line(df):
    window = 20   # 20 period 가 가장 무난하다고 한다

    # 20 줄 이상이여야함
    assert (len(df) >= 20)


    # Compute variables for bolinger
    ma = df["Price"].mean()
    typicalP = (df["High"] + df["Low"] + df["Price"])/3
    sd = typicalP.std()


    # compute variables for rsi
    delta = (df["Price"]).diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    rs = (up.ewm(span=14).mean()) / (down.abs().ewm(span=14).mean())  # rs based on ema

    # Inputs
    time = (df["Time"]).tail(1)
    price = (df["Price"]).tail(1)
    bolinger = ((price - ma) / sd)
    rsi = (100.0 - (100.0 / (1.0 + rs))).tail(1)
    vol = (df["Volume"]).tail(1)

    line = pd.concat([time, price, bolinger, rsi, vol], axis = 1, sort = False)
    line.columns = ['Time', 'Price', 'Bolinger', 'Rsi', 'Volume']
    return line

# train_line = (1 x 30)
# colums are : [b1, b2, ..., b10, r1, r2, ..., r10, v1, v2, ..., v10]
# 지난 10 period 동안 bolinger band index (b), Rsi (r), 거래량 (v)
# 10 period 동안의 상대적 value를 위해 normalized
def make_train_line(df):
    bol = (df["Bolinger"].tail(10)).to_numpy()
    bol = bol / np.linalg.norm(bol)
    rsi = (df["Rsi"].tail(10)).to_numpy()
    rsi = rsi / np.linalg.norm(rsi)
    vol = (df["Volume"].tail(10)).to_numpy()
    vol = vol / np.linalg.norm(vol)
    state = np.concatenate((bol, rsi, vol))
    return state