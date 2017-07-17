import colorsys
import functools
import pandas as pd
import datetime as dt
from dateutil.parser import parse
import numpy as np


class Instrument(object):
    def __init__(self, df, sample_period, timesofday):
        self._raw_data = df
        self.timesofday = timesofday
        self.sample_period = sample_period  # sample period for vol estimation in minutes
        self.id = df["InstrumentID"][0]
        self.market_time, self.hour = self.get_market_time()
        self.clean_data = self.clean_data()
        self.lags = [6] # best lag is 6
        
    def clean_data(self):
        """Clean dataframe"""
        df = self._raw_data
        df["UpdateTime"] = df["UpdateTime"].apply(lambda x: parse(x).time())
        df["period"] = df["UpdateTime"].apply(lambda x: (x.hour*60 + x.minute) // self.sample_period)
        # select data within makert time
        mask = pd.Series(np.zeros(len(df))).astype(bool)
        for period in self.market_time:
            if period[0] < period[1]:
                mask = np.logical_or(mask, np.logical_and(df["UpdateTime"] >= period[0], df["UpdateTime"] <= period[1]))
            else:
                mask = np.logical_or(mask, np.logical_or(df["UpdateTime"] >= period[0], df["UpdateTime"] <= period[1]))
        # filter outliers
        #mask = np.logical_and(mask, ((df["LastPrice"] - df["LastPrice"].mean()) / df["LastPrice"].std()).abs() < 10)
        mask = np.logical_and(mask, df["LastPrice"] > 0)
        df = df[mask].copy()
        df["LogPrice"] = df["LastPrice"].apply(np.log)
        return df
        
    def get_market_time(self):
        """Create a dataframe with maket open periods"""
        df = pd.DataFrame(index = ["ag", "bu", "rb", "ru", "zn"],
                          columns = ["day_period_1", "day_period_2", "day_period_3", "night_period_1"])
        df.index.name = "Instrument"
        df.columns.name = "open periods"
        for row in df.index:
            df.loc[row]["day_period_1"] = (dt.time(9, 0, 0), dt.time(10, 14, 59, 99999))      # avoid the last datapoint
            df.loc[row]["day_period_2"] = (dt.time(10, 30, 0), dt.time(11, 29, 59, 99999))
            df.loc[row]["day_period_3"] = (dt.time(13, 30, 0), dt.time(14, 59, 59, 99999)) 
        for row in "ag", :
            df.loc[row]["night_period_1"] = (dt.time(21, 0, 0), dt.time(2, 29, 59, 99999))
        for row in "bu", "rb", "zn":
            df.loc[row]["night_period_1"] = (dt.time(21, 0, 0), dt.time(0, 59, 59, 99999))
        for row in "ru", :
            df.loc[row]["night_period_1"] = (dt.time(21, 0, 0), dt.time(10, 59, 59, 99999))
        market_time = df.loc[self.id[:2]]
        if self.timesofday == "day":
            market_time = market_time[:3]
        else:
            market_time = market_time[-1:]
        hour = 0
        for period in market_time:
            dum = (period[1].hour - period[0].hour) + (period[1].minute - period[0].minute) / 60.0
            hour += dum if dum > 0 else (dum + 24)
        return market_time, hour
    
    def get_vol(self):
        """Calculate volatility for the data"""
        vol_df = pd.DataFrame()
        if not self.clean_data.empty:
            annual_coef = np.sqrt(self.hour * 60 / self.sample_period * 252)
            grouped = self.clean_data.groupby("period")
            for lag in self.lags:
                vol_partial = functools.partial(self.vol_Zhou, annual_coef=annual_coef, k=lag)
                vol_df['vol_' + repr(lag)] = grouped["LogPrice"].apply(vol_partial)
            vol_df["timesofday"] = self.timesofday
            vol_df["instrument_id"] = self.id
            vol_df["datetime"] = grouped["Date"].first()
            vol_df["datetime"] = pd.to_datetime(vol_df["datetime"])
            vol_df.reset_index(inplace=True)
            vol_df["datetime"] += vol_df["period"].apply(lambda x: dt.timedelta(minutes=x * self.sample_period))
            vol_df.sort_values(by="datetime", inplace=True, kind="mergesort")
            self._reset_cache()
        return vol_df
        
    @staticmethod
    def vol_Zhou(logprice, annual_coef=1, k=1):
        """Calculate volatility of the series
        Inputs:
                logprice: series, LogPrice
                annual_coef: float, coefficient to annulize vol
                k: int, lag used to calculate volatility
        Outputs:
                vol: float, annuallized volatility
        """
        x = (logprice - logprice.shift(k)).dropna()
        vol_squared = (np.sum(x * x) + np.sum((x * x.shift(-k)).dropna()) + np.sum((x * x.shift(k)).dropna())) / k
        vol_squared = max(0, vol_squared)      # vol_squared could be negative 
        vol = np.sqrt(vol_squared) * annual_coef
        return vol
    
    def vol_tsrv(self, j=1, k=3):
        """Calculate volatility using TSVR AA formulat by LanZhang.
        DOESN'T PRODUCE CONSISTENT RESULT.
        """
        if j >= k:
            raise ValueError("j is not less than k.")
        s = self.clean_data["LogPrice"]
        n = len(s)
        n_k = (n - k + 1.) / k
        n_j = (n - j + 1.) / j
        vol_squared = (s.shift(-k) - s).dropna().apply(np.square).sum() - \
                    n_k / n_j * (s.shift(-j) - s).dropna().apply(np.square).sum()
        #vol_squared *= n / ((k - j) * n_k)
        vol = np.sqrt(vol_squared / self.hour * 24 * 252)
        return vol
    
    def _reset_cache(self):
        self._raw_data = None
        self.clean_data = None