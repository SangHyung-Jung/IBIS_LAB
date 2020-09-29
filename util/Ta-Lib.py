'''
얘는 보조지표 (Tech indi) 가져오는 py인데
만약에 너가 path같은거 전부 맞춰서 실행해보면서 할 생각이라면
여기는 path 정도만 수정해도 될듯?
딱히 건드릴꺼는 없어
'''


import talib
from talib import MA_Type
import os
import numpy as np
import pandas as pd

load_path = "/home/ir1067/price_data/"
save_path = "/home/ir1067/price_w_indicator/"
df = pd.read_csv('/home/ir1067/data/kospi.csv')

def add_tech_stats(company):

    df = pd.read_excel(load_path + company + '.xlsx', encoding = 'cp949', index_col=0).astype('double')
    
    open = df['open']
    close = df['close']
    adj_close = df['adj_close']
    volume = df['volume']
    
    # rsi
    rsi14 = talib.RSI(np.asarray(df['adj_close']), 14)

    # MACD
    macd, macdsignal, macdhist = talib.MACD(np.asarray(df['adj_close']), 12, 26, 9)

    # CCI
    cci = talib.CCI( df['high'], df['low'], df['adj_close'], timeperiod = 14 )

    # ADX
    adx = talib.ADX( df['high'], df['low'], df['adj_close'], timeperiod = 14 )

    # STOCH
    slowk, slowd = talib.STOCH (df['high'], df['low'], df['adj_close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

    # Williams' %R
    willr = talib.WILLR(df['high'], df['low'], df['adj_close'], timeperiod=14)

    # Momentum
    momentum = talib.MOM(df['adj_close'], 10)

    # Rate of change
    roc = talib.ROC(df['adj_close'], timeperiod = 10)

    # EMA
    ema8 = talib.EMA(df['adj_close'], timeperiod = 8)
    ema20 = talib.EMA(df['adj_close'], timeperiod = 20)
    ema200 = talib.EMA(df['adj_close'], timeperiod = 200)

    # StockRsi
    fastk, fastd = talib.STOCHRSI(df['adj_close'], 14, 3, 3)

    # Accumulation/Distribution (A/D) Oscillator
    adosc = talib.ADOSC(df['high'], df['low'], df['adj_close'], df['volume'], fastperiod=3, slowperiod=10)

    # On Balance Volume
    obv = talib.OBV(df['adj_close'], df['volume'])

    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(df['adj_close'], matype = MA_Type.T3)

    tmp = pd.DataFrame([rsi14, macd, cci, adx, slowk, slowd, willr, momentum, roc,
                                     ema8, ema20, ema200, fastk, fastd, adosc, obv, upper, middle, lower]).T

    tmp.columns = ['rsi_14', 'macd','cci', 'adx', 'stoch_slowk', 'stoch_slowd', 'willr', 'momentum', 'roc', 'ema8', 'ema20', 'ema200',
       'Sto_rsi_fastk', 'Sto_rsi_fastd', 'adosc', 'obv', 'BB_upper', 'BB_middle', 'BB_lower']
    tmp.index = df.index

    df = pd.concat([df, tmp], axis=1)

    df.to_csv(save_path + company + '.csv')

for company in os.listdir(load_path):
    if company == "유한양행.xlsx":
        pass
    else:
        continue
    print(company + 'start')
    add_tech_stats(company[:-5])


#https://mrjbq7.github.io/ta-lib/