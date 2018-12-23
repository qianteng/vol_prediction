import config
import os
import functools
import datetime as dt
import pdb
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook      #tqdm
from sklearn.preprocessing import StandardScaler

# test
class Instrument(object):
    def __init__(self, df, sample_period, timesofday):
        """Class to clean data and calculate volatility for a instrument

        Args:
            df: dataframe, a single day data
            sample_period: int, sample period for vol estimation in minutes
            timesofday: str, 'day' or 'night'
        Returns:
            None
        """
        self._raw_data = df
        self.timesofday = timesofday
        self.sample_period = sample_period
        self.id = df["InstrumentID"][0]
        self.market_time, self.hour = self.get_market_time()
        self.clean_data = self._get_clean_data()
        self.lags = [config.LAG]

    def _get_clean_data(self):
        """Clean dataframe

        Args:
            None
        Returns:
            df: DataFrame, clean data
        """
        df = self._raw_data
        df["UpdateTime"] = df["Date_UpdateTime"].apply(lambda x: x.time())
        df["Period"] = df["UpdateTime"].apply(lambda x: (x.hour*60 + x.minute) // self.sample_period)
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
        """Create a dataframe with trading time information

        Args:
            None
        Returns:
            market_time: dict, trading time of the instrument
            hour: float, total trading time within a day in hours
        """
        df = pd.DataFrame(index=["ag", "bu", "rb", "ru", "zn"],
                          columns=["DayPeriod1", "DayPeriod2", "DayPeriod3", "NightPeriod1"])
        df.index.name = "Instrument"
        df.columns.name = "open periods"
        for row in df.index:
            df.loc[row]["DayPeriod1"] = (dt.time(9, 0, 0), dt.time(10, 14, 59, 99999))      # avoid the last datapoint
            df.loc[row]["DayPeriod2"] = (dt.time(10, 30, 0), dt.time(11, 29, 59, 99999))
            df.loc[row]["DayPeriod3"] = (dt.time(13, 30, 0), dt.time(14, 59, 59, 99999))
        for row in ["ag"]:
            df.loc[row]["NightPeriod1"] = (dt.time(21, 0, 0), dt.time(2, 29, 59, 99999))
        for row in "bu", "rb", "zn":
            df.loc[row]["NightPeriod1"] = (dt.time(21, 0, 0), dt.time(0, 59, 59, 99999))
        for row in ["ru"]:
            df.loc[row]["NightPeriod1"] = (dt.time(21, 0, 0), dt.time(10, 59, 59, 99999))
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
        """Calculate volatility for the instrument

        Args:
            None
        Return:
            vol_df: DataFrame, volatility and micromarket features to predict volatility
        """
        vol_df = pd.DataFrame()
        if not self.clean_data.empty:
            annual_coef = np.sqrt(self.hour * 60 / self.sample_period * 252)
            grouped = self.clean_data.groupby("Period")
            for lag in self.lags:
                vol_partial = functools.partial(self.vol_Zhou, annual_coef=annual_coef, k=lag)
                vol_df['Vol_' + repr(lag)] = grouped["LogPrice"].apply(vol_partial)
                vol_df['Return'] = grouped["LogPrice"].last().diff().fillna(0) * annual_coef**2
                vol_df['BidAskImbal'] = grouped.apply(lambda x: x['AskVolume1'].sub(x['BidVolume1']).abs().mean())
                vol_df['Volume'] = grouped['Volume'].mean()
                vol_df['Volume_BAI_Ratio'] = vol_df['Volume'].div(vol_df['BidAskImbal'].clip_lower(0.01))  # bad
                vol_df['Spread'] = grouped.apply(lambda x: x['AskPrice1'].sub(x['BidPrice1']).abs().mean())
                trade_price = grouped.apply(lambda x: x['AskPrice1'].add(x['BidPrice1']).div(2).mean())
                mid_price = grouped['AveragePrice'].mean()
                vol_df['Trade_Mid_Ratio'] = trade_price.div(mid_price).replace(np.inf, 1)  # trade price/mid price, bad
                vol_df['HighLow'] = grouped.apply(lambda x: x["HighPrice"].sub(x["LowPrice"]).abs().mean())
                vol_df['Turnover'] = grouped["Turnover"].mean()
            vol_df["Timesofday"] = self.timesofday
            vol_df["InstrumentId"] = self.id
            vol_df["Datetime"] = grouped["Date"].first()
            vol_df["Datetime"] = pd.to_datetime(vol_df["Datetime"])
            vol_df.reset_index(inplace=True)
            vol_df["Datetime"] += vol_df["Period"].apply(lambda x: dt.timedelta(minutes=x * self.sample_period))
            vol_df.index = range(len(vol_df))
            self._reset_cache()
        return vol_df

    @staticmethod
    def vol_Zhou(logprice, annual_coef=1, k=1):
        """Calculate volatility of the series with Zhou Bin's method
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

    # def vol_tsrv(self, j=1, k=3):
    #     """Calculate volatility using TSVR AA formulat by LanZhang.
    #     DOESN'T PRODUCE CONSISTENT RESULT.
    #     """
    #     if j >= k:
    #         raise ValueError("j is not less than k.")
    #     s = self.clean_data["LogPrice"]
    #     n = len(s)
    #     n_k = (n - k + 1.) / k
    #     n_j = (n - j + 1.) / j
    #     vol_squared = (s.shift(-k) - s).dropna().apply(np.square).sum() - \
    #                 n_k / n_j * (s.shift(-j) - s).dropna().apply(np.square).sum()
    #     #vol_squared *= n / ((k - j) * n_k)
    #     vol = np.sqrt(vol_squared / self.hour * 24 * 252)
    #     return vol

    def _reset_cache(self):
        self._raw_data = None
        self.clean_data = None


def calculate_vol_one(file_paths, sample_period=5):
    """calculate volatility for one file and save in file

    Args:
        file_paths: str, path of the file
        sample_period: int, sample period used to calculate vol in minutes
    Returns:
        None
    """
    vols = []
    for file_path in tqdm_notebook(file_paths, desc='date loop'):
        df = pd.read_csv(file_path, parse_dates=[[1, 36]], keep_date_col=True)
        timesofday = file_path.split('/')[-2]
        if not df.empty:
            inst = Instrument(df, sample_period, timesofday)
            vols.append(inst.get_vol())
        else:
            print(file_path)
    vols_df = pd.concat(vols)
    vols_df.sort_values(by="Datetime", inplace=True, kind="mergesort")
    vols_df.index = range(len(vols_df))
    vols_df.to_csv(os.path.join(config.OUTPUT_DATA_PATH, inst.id + '.csv'))


def preprocess(InstrumentId):
    """Preprocess volatility data for prediction. Standardize raw features with a expanding window and drop columns that's not needed

    Args:
        df: DataFrame, raw data with volatility and feautres.
    Returns:
        df_std: DataFrame, standardized features.
    """
    df_full = pd.read_csv(os.path.join(config.OUTPUT_DATA_PATH, InstrumentId + '.csv'), parse_dates=['Datetime'], index_col=0)
    cols = ['Return', 'BidAskImbal', 'Volume', 'Spread', 'HighLow', 'Turnover']
    df = df_full[cols]
    df_std = df.copy()
    scaler = StandardScaler()
    min_periods = config.ROLLING_WINDOW
    df_std.iloc[:min_periods, :] = scaler.fit_transform(df.iloc[:min_periods, :])
    for i in range(min_periods, len(df)):
        scaler.fit(df.iloc[:i, :])
        df_std.iloc[i, :] = scaler.transform(df_std.iloc[i, :].values.reshape(1, -1))
    df_std[config.VOL_NAME] = df_full[config.VOL_NAME]
    df_std.to_csv(os.path.join(config.OUTPUT_DATA_PATH, InstrumentId + '_std.csv'))


def calculate_vol(ids):
    """Calculate, combine and preprocess volatility for an instrument

    Args:
        ids: list, instrument ids to calculate vol.
    """
    commodities = [name for name in os.listdir(config.DATA_BASE_PATH) if not name.startswith(".")]
    timesofdays = ["day", "night"]

    for commodity in commodities[:1]:
        file_paths = []
        file_paths_dict = defaultdict(list)
        for timesofday in timesofdays:
            path = os.path.join(config.DATA_BASE_PATH, commodity, timesofday)
            file_paths += [os.path.join(path, filename) for filename in os.listdir(path)]
        for file_path in file_paths:
            instrument_id = os.path.basename(file_path).split('_')[0]
            file_paths_dict[instrument_id].append(file_path)
        for instrument_id in tqdm_notebook(ids, desc='InstrumentId loop'):
        # for instrument_id in tqdm_notebook(file_paths_dict.keys()):
            calculate_vol_one(file_paths_dict[instrument_id])
            preprocess(instrument_id)






















