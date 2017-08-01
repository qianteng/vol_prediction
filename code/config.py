# Globle variables of configuration parameters


DATA_BASE_PATH = "/Volumes/WorkDisk/data/tick2016"                          # path of raw data
OUTPUT_PATH = "/Users/qt/Desktop/Notes/vol_prediction/output"               # path to save figures and outputs
OUTPUT_DATA_PATH = "/Users/qt/Desktop/Notes/vol_prediction/output_data"     # path to save calculated volatility
ROLLING_WINDOW = 120                                                        # rolling window used in predicting volatility
LAG = 6                                                                     # best lag is 6
VOL_NAME = 'Vol_' + repr(LAG)                                               # name of calculated volatility in the Dataframe
