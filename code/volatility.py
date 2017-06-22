import config
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import ipdb


def clean_data(df):
    """Clean dataframe
    """

    UTC8 = dt.timezone(dt.timedelta(hours = 8))
    df["DateTime"] = df["TimeStamp"].apply(lambda x: dt.datetime.fromtimestamp(int(x)/1000, tz = UTC8))
    df["Date"] = df["DateTime"].apply(lambda x: x.date())
    df["UpdateTime"] = df["DateTime"].apply(lambda x: x.time())
    return df

def daily_vol(df, k = 1):
    """Calculate daily volatility for dataframe.
    """
    trading_times = [(dt.time(hour = 9, minute = 0, second = 0), dt.time(hour = 10, minute = 15, second = 0)),
    (dt.time(hour = 10, minute = 30, second = 0), dt.time(hour = 11, minute = 30, second = 0)),
    (dt.time(hour = 13, minute = 30, second = 0), dt.time(hour = 15, minute = 0, second = 0)),
    ]
    vol_sq = 0
    for trading_time in trading_times:
        trading_df = timesofday_df[np.logical_and(timesofday_df["UpdateTime"] >= trading_time[0], timesofday_df["UpdateTime"] <= trading_time[1])]
        vol_sq = vol_sq + vol_squared(trading_df, k)
    vol = np.sqrt(vol_sq /3.75 * 24 * 252)
    return vol

def vol_squared(df, k = 1):
    """Calculate squre of volatility for dataframe, the prices in df is assumed to be continuous in time
    Inputs:
            df: dataframe, raw data of commodity futures price in a day
            k: int, lag used to calculate volatility
    Outputs:
            vol_squared: float, volatility of the dataframe
    """

    s = df["LastPrice"].apply(np.log)
    x = (s - s.shift(k)).dropna()
    vol_squared = (np.sum(x * x) + np.sum((x * x.shift(-k)).dropna()) + np.sum((x * x.shift(k)).dropna())) / k
    return vol_squared
        
commodities = [name for name in os.listdir(config.DATA_BASE_PATH) if not name.startswith(".")] 
timesofdays = ["day", "night"]
opt_lag_max = []
opt_lag_min = []
for commodity in commodities[0], :
    for timesofday in timesofdays[0], :
        timesofday_path = os.path.join(config.DATA_BASE_PATH, commodity, timesofday)
        timesofday_files = os.listdir(timesofday_path)
        for timesofday_file in timesofday_files[:5] :
            timesofday_df = pd.read_csv(os.path.join(timesofday_path, timesofday_file))
            timesofday_df = clean_data(timesofday_df)
            lags = range(1, 101, 1)
            vols = []
            for k in lags:
                vols.append(daily_vol(timesofday_df, k = k))
            opt_lag_max.append(lags[np.argmax(vols)])
            opt_lag_min.append(lags[np.argmin(vols)])
            plt.plot(lags, vols)
        plt.xlabel("lag")
        plt.ylabel("vol")
        plt.title("vol estimation with different lags")
        plt.savefig(os.path.join(config.OUTPUT_PATH, "vol_lag.png"), dpi = 1000)
        plt.close()


