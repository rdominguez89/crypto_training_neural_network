import warnings
from pandas.errors import PerformanceWarning
import pandas as pd
from numba import njit
from run_training_loop_torch import run_neural
from run_training_loop_torch_one import run_neural_one
import numpy as np
warnings.simplefilter("ignore", PerformanceWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*DataFrame is highly fragmented.*")
warnings.filterwarnings("ignore", "The `backend` parameter is set to `cuda`.*")

"""
features.py

This module contains feature extraction functions for the project.
Add new feature functions below. Each function should compute a single feature.
"""
def add_ny_time_features(df, features_columns, use_NY_trading_hour, use_day_month):
    """Add New York time, trading hour, day, and month features."""
    if use_NY_trading_hour or use_day_month:
        df['date_ny'] = df['date'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
        def is_ny_trading_hour(dt):
            if dt.weekday() >= 5:
                return 0
            hour = dt.hour
            minute = dt.minute
            if (hour > 9 and hour < 16):
                return 1
            if hour == 9 and minute >= 30:
                return 1
            if hour == 16 and minute == 0:
                return 0
            return 0
        if use_NY_trading_hour:
            df['ny_trading_hour'] = df['date_ny'].apply(is_ny_trading_hour)
            features_columns.append('ny_trading_hour')
    if use_day_month == 'day' or use_day_month == 'day_month':
        df['day'] = df['date_ny'].dt.weekday
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 6)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 6)
        features_columns += ['day_sin', 'day_cos']
    if use_day_month == 'month' or use_day_month == 'day_month':
        df['month'] = df['date_ny'].dt.month - 1
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 11)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 11)
        features_columns += ['month_sin', 'month_cos']
    return df, features_columns

def get_columns_ma(ma_periods):
    return [f'above_MA_{period_ma}' for period_ma in ma_periods]

def get_columns_green():
    return ['green']

def get_columns_vwap(ma_periods):
    return [f'above_vwap_{period_ma}' for period_ma in ma_periods]

def get_columns_range(periods_look_back):
    columns_range = []
    for col_name in ['high_low', 'open_close', 'volume']:
        for period_look_back in periods_look_back:
            columns_range.extend([f'{col_name}_{stat}_range_{period_look_back}' for stat in ['avg', 'max', 'min', 'std']])
    return columns_range


def do_ma_feature_engineering(df, ma_periods, periods_look_back, use_NY_trading_hour=False, use_day_month=None):
    """Main function to add all features and return columns lists."""
    features_columns = []
    df, features_columns = add_ny_time_features(df, features_columns, use_NY_trading_hour, use_day_month)
    df, features_columns = add_moving_averages(df, features_columns, ma_periods)
    df, features_columns = add_vwap_features(df, features_columns, ma_periods)
    df, features_columns = add_diff_features(df, features_columns, method='raw') 
    df, features_columns = add_range_features(df, features_columns, periods_look_back)

    df, features_columns = add_advanced_features(df, features_columns, ma_periods)
    df, features_columns = add_atr_rolling_std(df, features_columns)
    df, features_columns = add_normalized_features(df, features_columns, ma_periods)
    df, features_columns = add_trend_volatility_features(df, features_columns)
    df, features_columns = add_volume_features(df, features_columns)
    df, features_columns = add_candle_shape_features(df, features_columns)
    df, features_columns = add_event_features(df, features_columns, ma_periods[0])
    return df, features_columns

def do_search_entries(side, columns_interest, points_all, ratio, df, df_1m, fract_ratio=0.6, range_low_limit=130, range_top_limit=2000, shift_open=0.0):
    # Convert series to numpy arrays for numba processing.
    open_prices = df['open'].to_numpy()
    low_prices = df['low'].to_numpy()
    high_prices = df['high'].to_numpy()
    low_prices_1m = df_1m['low'].to_numpy()
    high_prices_1m = df_1m['high'].to_numpy()
    dates = df['date'].to_numpy()
    dates_1m = df_1m['date'].to_numpy()

    @njit
    def search_long_entries(dates, open_prices, low_prices, high_prices, dates_1m, low_prices_1m, high_prices_1m, points_all, ratio, fract_ratio, range_low_limit, range_top_limit, shift_open):
        n = len(open_prices) 
        # Initialize with -1 to mark unassigned entries.
        long_entries = -1 * np.ones(n, dtype=np.int64)
        for i in range(n-1):
            points = points_all[i]
            if points < range_low_limit: continue
            if points > range_top_limit: continue
            op = open_prices[i+1] - shift_open*points
            sl = op - points
            tp = op + (points * fract_ratio*ratio)
            filled = False
            for j in range(i+1, min([n, i + 200])): # Limit lookahead to 200 candles
                if not filled:
                    if  high_prices[j] > tp:
                        long_entries[i] = 0 #added as test
                        break
                    if low_prices[j] < op:
                        filled = True
                #if not filled: break
                if filled:
                    if low_prices[j] < sl and high_prices[j] > tp: 
                        #break
                        try:
                            j1m_1, j1m_2 = np.where((dates_1m == dates[j]) | (dates_1m == dates[j+1]))[0]
                        except:
                            print('Error in finding 1m dates')
                            break
                        for j1 in range(j1m_1,j1m_2):
                            if low_prices_1m[j1] < sl and high_prices_1m[j1] > tp: break
                            if low_prices_1m[j1] < sl:
                                long_entries[i] = 0 # No entry
                                break
                            elif high_prices_1m[j1] > tp:
                                long_entries[i] = 1 # Entry found
                                break
                        break
                    if low_prices[j] < sl:
                        long_entries[i] = 0 # No entry
                        break
                    elif high_prices[j] > tp:
                        long_entries[i] = 1 # Entry found
                        break
        return long_entries

    @njit
    def search_short_entries(dates, open_prices, low_prices, high_prices, dates_1m, low_prices_1m, high_prices_1m, points_all, ratio, fract_ratio, range_low_limit, range_top_limit, shift_open):
        n = len(open_prices)
        short_entries = -1 * np.ones(n, dtype=np.int64)
        for i in range(n - 1):
            points = points_all[i-1]
            if points < range_low_limit: continue
            if points > range_top_limit: continue
            op = open_prices[i] + shift_open*points
            tp = op - (points * fract_ratio*ratio)
            sl = op + points
            filled = False
            for j in range(i, min([n, i + 200])): # Limit lookahead to 200 candles
                if not filled:
                    if low_prices[j] < tp:
                        short_entries[i] = 0 #added as test
                        break
                    if high_prices[j] > op:
                        filled = True
                #if not filled: break
                if filled:
                    if low_prices[j] < tp and high_prices[j] > sl:
                        #break
                        try:
                            j1m_1, j1m_2 = np.where((dates_1m == dates[j]) | (dates_1m == dates[j+1]))[0]
                        except:
                            print('Error in finding 1m dates')
                            break
                        for j1 in range(j1m_1,j1m_2):
                            if low_prices_1m[j1] < tp and high_prices_1m[j1] > sl: break
                            if low_prices_1m[j1] < tp:
                                short_entries[i] = 1 # Entry found
                                break
                            elif high_prices_1m[j1] > sl:
                                short_entries[i] = 0 # No entry
                                break                    
                        break
                    if low_prices[j] < tp:
                        short_entries[i] = 1 # Entry found
                        break
                    elif high_prices[j] > sl:
                        short_entries[i] = 0 # No entry
                        break
        return short_entries

    # Apply the optimized search routines separately.
    if side == 'long':
        df[f'long_entry'] = search_long_entries(dates, open_prices, low_prices, high_prices, dates_1m, low_prices_1m, high_prices_1m, points_all, ratio, fract_ratio, range_low_limit, range_top_limit, shift_open)
    elif side == 'short':
        df['short_entry'] = search_short_entries(dates, open_prices, low_prices, high_prices, dates_1m, low_prices_1m, high_prices_1m, points_all, ratio, fract_ratio, range_low_limit, range_top_limit, shift_open)
    else:
        raise ValueError("Invalid side. Must be 'long' or 'short'.")
    columns_interest += [f'{side}_entry']
    # Keep only relevant columns to save memory
    df = df[columns_interest]
    return df, columns_interest


@njit
def build_features(data, window_size, i_start):
    n, f = data.shape
    features = np.empty((n, window_size * f), dtype=np.float64)
    for i in range(i_start, n):
        # Take a window of shape (window_size, f), flatten to (window_size*f,)
        features[i] = data[i - window_size+1:i+1, :].flatten()
    return features
    
def do_reshape_window(df, window_size, ma_periods, df_ori, columns_features, columns_for_windows):
    for feature in columns_for_windows:
        if feature == 'ma': cols = get_columns_ma(ma_periods)
        if feature == 'vwap': cols = get_columns_vwap(ma_periods)
        if feature == 'green': cols = get_columns_green()
        data = df_ori[cols].values.astype(int)
        features = build_features(data, window_size, ma_periods[-1] + window_size - 1)
        feature_names = []
        for t in reversed(range(window_size)):
            for col in cols:
                feature_names.append(f"{feature}_{col}_t{t}")
        df[feature_names] = features
        columns_features.extend(feature_names)
    
    return df

def do_training(df, side, ratio, timeframe, frac, limit_low_range, limit_top_range, shift_open, use_NY_trading_hour, use_day_month, method, prices_col_1, prices_col_2, period_target, std_dev, multiplier, window, ma_periods, periods_look_back):
    df = df.dropna().reset_index(drop=True)
    run_neural(df, side, ratio, [], timeframe, frac, limit_low_range, limit_top_range, shift_open, use_NY_trading_hour, use_day_month, method, prices_col_1, prices_col_2, period_target, std_dev, multiplier, window, ma_periods, periods_look_back)

def do_training_one(df, side, ratio, timeframe, frac, limit_low_range, limit_top_range, shift_open, use_NY_trading_hour, use_day_month, method, prices_col_1, prices_col_2, period_target, std_dev, multiplier, window, ma_periods, periods_look_back, neurons):
    df = df.dropna().reset_index(drop=True)
    run_neural_one(df, side, ratio, [], timeframe, frac, limit_low_range, limit_top_range, shift_open, use_NY_trading_hour, use_day_month, method, prices_col_1, prices_col_2, period_target, std_dev, multiplier, window, ma_periods, periods_look_back, neurons)

def read_file(timeframe):
    df = pd.read_csv(f'../test_5/history_BTCUSDT_{timeframe}_10_2021_3_9_2025.csv')
    df_1m = pd.read_csv(f'../test_5/history_BTCUSDT_1m_10_2021_3_9_2025.csv')
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df_1m['date'] = pd.to_datetime(df_1m['date'], unit='ms')
    df = df[df['date'] < '2025-07-01']
    return df, df_1m

def add_moving_averages(df, features_columns, ma_periods):
    """Add moving average and above_MA features to df."""
    for period_ma in ma_periods:
        df[f'MA_{period_ma}'] = df['close'].rolling(window=period_ma, min_periods=period_ma).mean()
        df[f'above_MA_{period_ma}'] = (df['close'] > df[f'MA_{period_ma}']).astype(int)
        #features_columns.append(f'above_MA_{period_ma}')
    return df, features_columns

def add_vwap_features(df, features_columns, ma_periods):
    """Add VWAP and above_vwap features to df."""
    for period_ma in ma_periods:
        df[f'vwap_{period_ma}'] = (df['volume'] * df['close']).rolling(window=period_ma, min_periods=period_ma).sum() / df['volume'].rolling(window=period_ma, min_periods=period_ma).sum()
        df[f'above_vwap_{period_ma}'] = (df['close'] > df[f'vwap_{period_ma}']).astype(int)
        #features_columns.append(f'above_vwap_{period_ma}')
    return df, features_columns

def add_range_features(df, features_columns, periods_look_back):
    """Add range-based features (avg, max, min, std) for high-low, open-close, volume."""
    for col, col_name in zip(['diff_high_low', 'diff_open_close', 'volume'], ['high_low', 'open_close', 'volume']):
        for period_look_back in periods_look_back:
            df[f'{col_name}_avg_range_{period_look_back}'] = df[col].rolling(window=period_look_back).mean()
            df[f'{col_name}_max_range_{period_look_back}'] = df[col].rolling(window=period_look_back).max()
            df[f'{col_name}_min_range_{period_look_back}'] = df[col].rolling(window=period_look_back).min()
            df[f'{col_name}_std_range_{period_look_back}'] = df[col].rolling(window=period_look_back).std()
            features_columns += [f'{col_name}_avg_range_{period_look_back}', f'{col_name}_max_range_{period_look_back}', f'{col_name}_min_range_{period_look_back}', f'{col_name}_std_range_{period_look_back}']
    return df, features_columns

def add_diff_features(df, features_columns, method='abs'):
    """Add diff_high_low and diff_open_close columns."""
    df['diff_high_low'] = (df['high'] - df['low']).abs()
    df['diff_open_close'] = (df['open'] - df['close']).abs()
    if method == 'raw':
        df['green'] = (df['close'] > df['open']).astype(int)
        #features_columns += ['green']
    #features_columns += ['diff_high_low', 'diff_open_close']
    return df, features_columns

def add_advanced_features(df, features_columns, ma_periods):
    """Add distance to MA/VWAP and slope features."""
    for period_ma in ma_periods:
        df[f'dist_to_MA_{period_ma}'] = df['close'] - df[f'MA_{period_ma}']
        df[f'slope_MA_{period_ma}'] = df[f'MA_{period_ma}'].diff()
        df[f'dist_to_vwap_{period_ma}'] = df['close'] - df[f'vwap_{period_ma}']
        df[f'slope_vwap_{period_ma}'] = df[f'vwap_{period_ma}'].diff()
        features_columns += [f'dist_to_MA_{period_ma}', f'slope_MA_{period_ma}', f'dist_to_vwap_{period_ma}', f'slope_vwap_{period_ma}']
    return df, features_columns

def add_atr_rolling_std(df, features_columns, window=14):
    """Add ATR and rolling std features."""
    df['atr_14'] = (df['high'] - df['low']).rolling(window=window).mean()
    df['rolling_std_14'] = df['close'].rolling(window=window).std()
    features_columns += ['atr_14', 'rolling_std_14']
    return df, features_columns

def add_normalized_features(df, features_columns, ma_periods):
    """Add normalized distance and slope features."""
    for period_ma in ma_periods:
        df[f'norm_dist_to_MA_{period_ma}'] = df[f'dist_to_MA_{period_ma}'] / df['atr_14']
        df[f'norm_dist_to_vwap_{period_ma}'] = df[f'dist_to_vwap_{period_ma}'] / df['atr_14']
        df[f'norm_slope_MA_{period_ma}'] = df[f'slope_MA_{period_ma}'] / df['rolling_std_14']
        df[f'norm_slope_vwap_{period_ma}'] = df[f'slope_vwap_{period_ma}'] / df['rolling_std_14']
        features_columns += [f'norm_dist_to_MA_{period_ma}', f'norm_dist_to_vwap_{period_ma}', f'norm_slope_MA_{period_ma}', f'norm_slope_vwap_{period_ma}']
    return df, features_columns

def calc_adx(df, n=14):
    """Calculate ADX indicator."""
    high = df['high']
    low = df['low']
    close = df['close']
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr1 = pd.DataFrame({'h-l': high - low, 'h-c': (high - close.shift()), 'l-c': (low - close.shift())})
    tr = tr1.abs().max(axis=1)
    atr = tr.rolling(n).mean()
    plus_di = 100 * (plus_dm.rolling(n).mean() / atr)
    minus_di = 100 * (minus_dm.abs().rolling(n).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(n).mean()
    return adx

def add_trend_volatility_features(df, features_columns):
    """Add ADX, realized volatility, and volatility percentile features."""
    df['adx_14'] = calc_adx(df)
    df['realized_vol_14'] = np.log(df['close']).diff().rolling(window=14).std() * np.sqrt(252)
    df['vol_percentile_14'] = df['realized_vol_14'].rank(pct=True)
    features_columns += ['adx_14', 'realized_vol_14', 'vol_percentile_14']
    return df, features_columns

def add_volume_features(df, features_columns):
    """Add volume ratio and rvol features."""
    df['volume_ratio_14'] = df['volume'] / df['volume'].rolling(window=14).mean()
    df['rvol_14'] = (df['volume'] - df['volume'].rolling(window=14).mean()) / df['volume'].rolling(window=14).std()
    features_columns += ['volume_ratio_14', 'rvol_14']
    return df, features_columns

def add_candle_shape_features(df, features_columns):
    """Add body size, wick, and related ratio features."""
    df['body_size'] = (df['close'] - df['open']).abs()
    df['body_size_ratio'] = df['body_size'] / (df['high'] - df['low']).replace(0, np.nan)
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
    df['upper_wick_ratio'] = df['upper_wick'] / (df['high'] - df['low']).replace(0, np.nan)
    df['lower_wick_ratio'] = df['lower_wick'] / (df['high'] - df['low']).replace(0, np.nan)
    features_columns += ['body_size', 'body_size_ratio', 'upper_wick', 'lower_wick', 'upper_wick_ratio', 'lower_wick_ratio']
    return df, features_columns

def add_event_features(df, features_columns, ma_period):
    """Add new high/low, time since, bullish candle, and run length features."""
    df['new_high'] = (df['high'] == df['high'].cummax()).astype(int)
    df['new_low'] = (df['low'] == df['low'].cummin()).astype(int)
    df['time_since_new_high'] = (~df['new_high'].astype(bool)).groupby(df['new_high'].cumsum()).cumcount()
    df['time_since_new_low'] = (~df['new_low'].astype(bool)).groupby(df['new_low'].cumsum()).cumcount()
    df['bullish_candle'] = (df['close'] > df['open']).astype(int)
    df['bullish_count_14'] = df['bullish_candle'].rolling(window=14).sum()
    df['run_length_above_MA'] = df[f'above_MA_{ma_period}'].groupby((df[f'above_MA_{ma_period}'] != df[f'above_MA_{ma_period}'].shift()).cumsum()).cumcount() + 1
    df['run_length_below_MA'] = df['run_length_above_MA'].where(df[f'above_MA_{ma_period}'] == 0, 0)
    features_columns += ['new_high', 'new_low', 'time_since_new_high', 'time_since_new_low', 'bullish_candle', 'bullish_count_14', 'run_length_above_MA', 'run_length_below_MA']
    return df, features_columns
