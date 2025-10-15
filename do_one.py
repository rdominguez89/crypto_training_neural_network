import numpy as np
from datetime import datetime, timezone, timedelta
from features_calc_mod import do_search_entries, do_reshape_window, do_training, do_training_one, read_file, do_ma_feature_engineering

ma_periods, periods_look_back, windows = np.arange(5,201,5), [10, 30, 50, 70, 90, 120, 140, 160, 180, 200], range(4, 15, 2)

use_NY_trading_hour = True
use_day_month = 'None'  # Options: None, 'day', 'month
std_dev = 0
fract_ratio, range_low_limit, range_top_limit, shift_open = 0.6, 130, 2000, 0.6
side, timeframe, window, method, prices_col_1, prices_col_2, multiplier, period_target, ratio = 'long', '30m', 14, 'avg', 'open', 'close', 2, 30, 5  # Specify the timeframe you want to process

df, df_1m = read_file(timeframe)
print(f"Processing timeframe: {timeframe}, prices: {prices_col_1}-{prices_col_2}, method: {method}, std_dev: {std_dev}, multiplier: {multiplier}, shift_open: {shift_open}")
df, features_columns = do_ma_feature_engineering(df, ma_periods, periods_look_back, use_NY_trading_hour, use_day_month)
points_all = multiplier*(df[f'{prices_col_1}_{prices_col_2}_{method}_range_{period_target}'].to_numpy() + std_dev*df[f'{prices_col_1}_{prices_col_2}_std_range_{period_target}'].to_numpy())
print(f'Processing ratio: {ratio}')
df_1, features_columns_h = do_search_entries(side, features_columns.copy(), points_all, ratio, df.copy(), df_1m, fract_ratio, range_low_limit, range_top_limit, shift_open) # period to included column in df||can change to periods_look_back
df_all = do_reshape_window(df_1.copy(), window, ma_periods, df.copy(), features_columns_h, ['ma', 'vwap', 'green'])
df_side = df_all[df_all[f'{side}_entry']!=-1].reset_index(drop=True)
#do_training(df_side, side, ratio, timeframe, fract_ratio, range_low_limit, range_top_limit, shift_open, use_NY_trading_hour, use_day_month, method, prices_col_1, prices_col_2, period_target, std_dev, multiplier, window, ma_periods, periods_look_back)
do_training_one(df_side, side, ratio, timeframe, fract_ratio, range_low_limit, range_top_limit, shift_open, use_NY_trading_hour, use_day_month, method, prices_col_1, prices_col_2, period_target, std_dev, multiplier, window, ma_periods, periods_look_back, [256, 224, 192])

                                

print("All training completed.")
