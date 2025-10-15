import numpy as np
from datetime import datetime, timezone, timedelta
from features_calc import do_search_entries, do_reshape_window, do_training, read_file, do_ma_feature_engineering

min_rows = 5000
timeframes = ['30m']#,'1h','4h']#,'15m']  
do_long, do_short = True, False
ma_periods, periods_look_back, ratios, windows = np.arange(5,201,5), [10, 30, 50, 70, 90, 120, 140, 160, 180, 200], [2,3,4,5], range(4, 15, 2)

use_NY_trading_hour = True
use_day_month = 'None'  # Options: None, 'day', 'month
std_dev = 0
fract_ratio, range_low_limit, range_top_limit, shift_open = 0.6, 130, 2000, 0.7
for timeframe in timeframes:
    df, df_1m = read_file(timeframe)
    for method in ['avg','max']:
        for prices_col_1,  prices_col_2 in [('open', 'close'), ('high', 'low')]:
            for multiplier in [1,2]:
                print(f"Processing timeframe: {timeframe}, prices: {prices_col_1}-{prices_col_2}, method: {method}, std_dev: {std_dev}, multiplier: {multiplier}, shif_open: {shift_open}")
                df, features_columns = do_ma_feature_engineering(df, ma_periods, periods_look_back, use_NY_trading_hour, use_day_month)
                for k, period_target in enumerate(periods_look_back):
                    print(f"Processing {timeframe} progress: {(100*(k)/len(periods_look_back)):.2f}% {(datetime.now(timezone.utc) - timedelta(hours=4)).strftime('%Y-%m-%d %H:%M:%S')}")                    
                    points_all = multiplier*(df[f'{prices_col_1}_{prices_col_2}_{method}_range_{period_target}'].to_numpy() + std_dev*df[f'{prices_col_1}_{prices_col_2}_std_range_{period_target}'].to_numpy())
                    for ratio in ratios:
                        print(f'Processing ratio: {ratio}')
                        for side in ['long', 'short']:
                            df_1, features_columns_h = do_search_entries(side, features_columns.copy(), points_all, ratio, df.copy(), df_1m, fract_ratio, range_low_limit, range_top_limit, shift_open) # period to included column in df||can change to periods_look_back
                            for window in windows:
                                df_all = do_reshape_window(df_1.copy(), window, ma_periods, df.copy(), features_columns_h, ['ma', 'vwap', 'green'])
                                df_side = df_all[df_all[f'{side}_entry']!=-1].reset_index(drop=True)
                                if len(df_side) > min_rows:
                                    do_training(df_side, side, ratio, timeframe, fract_ratio, range_low_limit, range_top_limit, shift_open, use_NY_trading_hour, use_day_month, method, prices_col_1, prices_col_2, period_target, std_dev, multiplier, window, ma_periods, periods_look_back)
                print(f'Finished {timeframe}, prices: {prices_col_1}-{prices_col_2}, method: {method}, mx: {multiplier} {(datetime.now(timezone.utc) - timedelta(hours=3)).strftime('%Y-%m-%d %H:%M:%S')}')

print("All training completed.")
