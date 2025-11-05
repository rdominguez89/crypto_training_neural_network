import pandas as pd
import numpy as np
import warnings 
import torch
import joblib
import os

from pandas.errors import PerformanceWarning
warnings.simplefilter("ignore", PerformanceWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*DataFrame is highly fragmented.*")
from features_calc import do_ma_feature_engineering, do_reshape_window

device = 'cpu'

def get_inputs(model_name):

    # Remove the prefix
    model_name = model_name.replace('best_model_', '')

    # Split by underscores
    parts = model_name.split('_')



    # Parse the parts
    i = 0
    while i < len(parts):
        part = parts[i]
        
        if part in ['15m', '30m', '1h', '4h', '1d']:  # Add more timeframes as needed
            timeframe = part
        elif part in ['long', 'short']:
            side = part
        elif part == 'ratio':
            i += 1
            ratio = int(parts[i])
        elif part == 'window':
            i += 1
            window = int(parts[i])
        elif part == 'method':
            i += 1
            method = parts[i]
        elif part in ['open', 'high', 'low', 'close']:
            prices_col_1 = part
            if i + 1 < len(parts) and parts[i + 1] in ['open', 'high', 'low', 'close']:
                prices_col_2 = parts[i + 1]
                i += 1  # Skip next part since we used it
        elif part == 'mx':
            i += 1
            multiplier = int(parts[i])
        elif part == 'period':
            i += 1
            period_lb = int(parts[i])
        elif part == 'std':
            i += 1
            std_dev = int(parts[i])
        elif part == 'shift':
            i += 1
            shift_open = float(parts[i])
        elif part == 'balj':  # Stop parsing at balance/profit factor
            break
            
        i += 1
    return timeframe, side, ratio, window, method, prices_col_1, prices_col_2, multiplier, period_lb, std_dev, shift_open

def run_forward_test(model, ct, timeframe, frac_be, limit_low_range, limit_top_range, shift_open, use_NY_trading_hour, use_day_month, method, prices_col_1, prices_col_2, period_lb, std_dev, multiplier, ratio, side, window, ma_periods, periods_look_back, cols_original, X_train, do_month):
    tiny_profit = 0.0  # Fraction of range to set break-even stop loss
    print_log = False

    if do_month == 8:
        if timeframe == '15m':
            df = pd.read_csv(f'../test_5/history_BTCUSDT_{timeframe}_8_2025_20_8_2025.csv')
        elif timeframe == '30m':
            df = pd.read_csv(f'../test_5/history_BTCUSDT_{timeframe}_8_2025_23_8_2025.csv')
        df1m = pd.read_csv(f'../test_5/history_BTCUSDT_1m_8_2025_21_8_2025.csv')
    elif do_month == 7:
        df = pd.read_csv(f'../test_5/BTCUSDT-{timeframe}-2025-07.csv')
        df1m = pd.read_csv(f'../test_5/BTCUSDT-1m-2025-07.csv')
    elif do_month == 9:
        df = pd.read_csv(f'../test_5/BTCUSDT-{timeframe}-2025-09.csv')
        df1m = pd.read_csv(f'../test_5/BTCUSDT-1m-2025-09.csv')
    elif do_month == 't':
        df = pd.read_csv(f'../test_5/history_BTCUSDT_{timeframe}_10_2021_3_9_2025.csv')
        df1m = pd.read_csv(f'../test_5/history_BTCUSDT_1m_10_2021_3_9_2025.csv')
        df.rename(columns={'date': 'open_time'}, inplace=True)
        df1m.rename(columns={'date': 'open_time'}, inplace=True)
        df = df.iloc[:int(0.7*len(df))]
    try:
        df['date'] = pd.to_datetime(df['open_time'], unit='ms')
        df1m['date'] = pd.to_datetime(df1m['open_time'], unit='ms')
    except:
        df['date'] = pd.to_datetime(df['date'], unit='ms')
        df1m['date'] = pd.to_datetime(df1m['date'], unit='ms')

    df, features_columns = do_ma_feature_engineering(df, ma_periods, periods_look_back, use_NY_trading_hour, use_day_month)
    range_value_target = multiplier*(df[f'{prices_col_1}_{prices_col_2}_{method}_range_{period_lb}'].to_numpy() + std_dev*df[f'{prices_col_1}_{prices_col_2}_std_range_{period_lb}'].to_numpy())
    df_ori = df.copy()
    df = do_reshape_window(df.copy(), window, ma_periods, df_ori.copy(), features_columns, ['ma', 'vwap', 'green'])
    trading_hours = df['ny_trading_hour'].to_numpy()
    open_prices = df_ori['open'].to_numpy()
    close_prices = df_ori['close'].to_numpy()
    high = df_ori['high'].to_numpy()
    low = df_ori['low'].to_numpy()
    
 
    op_commission = 0.000#4  # Example commission rate
    sl_commission = 0.0005  # Example commission rate for stop loss
    tp_commission = 0.000#2  # Example commission rate for take profit

    val_result = []
    balance, n_loss, n_win, n_be, n_market, usd_win, usd_loss = 1000, 0, 0, 0, 0, 0, 0.0001
    balance_0 = balance
    percentage = 0.01
    in_position = False
    torch.cuda.empty_cache()
    model.to(device)

    X_scaled = ct.transform(df[cols_original])
    X_scaled_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        prediction = model(X_scaled_t)
        prob = (prediction.cpu() > 0.5).float()   

    for i in range(201, len(prob)):
        if i>=len(df)-1: break

        if not in_position:
            if range_value_target[i - 1] < limit_low_range or range_value_target[i - 1] > limit_top_range: continue   
            prediction_1 = int(prob[i-1] > 0.5)             # Make trading decision            
            if prediction_1 == 1:
                in_position = True
                range_value = range_value_target[i - 1]
                if side == 'long':  
                    op = round(open_prices[i] - shift_open * range_value,1)
                    tp, sl, be = round(op + ratio * range_value,1), round(op - range_value,1), round(op + frac_be*ratio*range_value,1)
                if side == 'short':
                    op = round(open_prices[i] + shift_open * range_value,1)
                    tp, sl, be = round(op - ratio * range_value,1), round(op + range_value,1), round(op - frac_be*ratio*range_value,1)
                i_open = i
                sl_ori = sl
                break_even = False
                size = round(balance * percentage / range_value, 3)
                filled = False
                if print_log:
                    print(f"Entering position")
                    print(f"Ticker: Date: {df_ori['date_ny'].iloc[i]}")
                    print(f"Ticker: Price: {op}")
                    print(f"Ticker: TP: {tp}")
                    print(f"Ticker: SL: {sl}")
                    print("")
            else:
                continue


        
        if in_position and not filled:
            if (side == 'long' and low[i] <= op and high[i] >= tp) or (side == 'short' and high[i] >= op and low[i] <= tp): #Check in 1m whether TP or OP was hit first
                df1m_h = df1m[(df1m['date'] >= df_ori['date'].iloc[i_open]) & (df1m['date'] < df_ori['date'].iloc[i+1])].reset_index(drop=True)
                for j in range(len(df1m_h)):
                    if (side == 'long' and df1m_h['high'].iloc[j] >= tp) or (side == 'short' and df1m_h['low'].iloc[j] <= tp):
                        in_position = False
                        if print_log:
                            print(f"Could not fill order after {i - i_open} candles, TP already hit, exiting.")
                            print(f"\nTicker: Balance: {balance}\n\n")
                        break
                    elif (side == 'long' and df1m_h['low'].iloc[j] <= op) or (side == 'short' and df1m_h['high'].iloc[j] >= op):
                        filled = True
                        i_filled = i
                        if print_log:
                            print(f"Order filled after {i - i_open} candles")
                            print(f"Ticker: Date: {df_ori['date_ny'].iloc[i]}")
                            print(f"Ticker: Price: {op}")
                            print("")
                        break

            elif (side == 'long' and low[i] <= op) or (side == 'short' and high[i] >= op):
                filled = True
                i_filled = i
                if print_log:
                    print(f"Order filled after {i - i_open} candles")
                    print(f"Ticker: Date: {df_ori['date_ny'].iloc[i]}")
                    print(f"Ticker: Price: {op}")
                    print("")
            elif i - i_open >= 200:
                in_position = False
                if print_log:
                    print(f"Could not fill order after {i - i_open} candles, exiting.")
                    print(f"\nTicker: Balance: {balance}\n\n")
            else:
                prediction_waiting = int(prob[i] > 0.5)
                if prediction_waiting == 0:
                    in_position = False
                    if print_log:
                        print(f"Cancelling order after {i - i_open} candles due to prediction change, exiting.")
                        print(f"\nTicker: Balance: {balance}\n\n")
                elif prediction_waiting == 1 and (limit_low_range < range_value_target[i] < range_value_target[i]) and ((side=='long' and round(close_prices[i] - shift_open * range_value_target[i],1) < op) or (side=='short' and round(close_prices[i] + shift_open * range_value_target[i],1) > op)):
                    in_position = False
                    if print_log:
                        print(f"Cancelling order after {i - i_open} candles due to current candle shows better entry price, exiting.")
                        print(f"\nTicker: Balance: {balance}\n\n")
                    continue
                elif (side=='long' and high[i]>=tp) or (side=='short' and low[i] <= tp):
                    in_position = False
                    if print_log:
                        print(f"Could not fill order after {i - i_open} candles, TP already hit, exiting.")
                        print(f"\nTicker: Balance: {balance}\n\n")
                continue




        if in_position and filled:
            if (side == 'long' and (high[i] >= tp or low[i] <= sl)) or (side == 'short' and (low[i] <= tp or high[i] >= sl)):
                in_position = False
                if print_log:
                    print(f"Exiting position after {i - i_open} candles")
                    print(f"Ticker: Date: {df_ori['date_ny'].iloc[i]}")
                winh=False
                if ((side == 'long' and (high[i] >= tp and low[i] <= sl)) or (side == 'short' and (low[i] <= tp and high[i] >= sl)) or i == i_filled):
                    df1m_h = df1m[(df1m['date'] >= df_ori['date'].iloc[i_filled])&(df1m['date'] < df_ori['date'].iloc[i+1])].reset_index(drop=True)
                    if break_even:
                        sl = sl_ori
                        break_even = False
                    filled_1m = False
                    for j in range(len(df1m_h)):
                        if not filled_1m and ((side == 'long' and df1m_h['low'].iloc[j] <= op) or (side == 'short' and df1m_h['high'].iloc[j] >= op)):
                            filled_1m = True
                        if filled_1m:
                            if ((side == 'long' and df1m_h['high'].iloc[j] >= tp) or (side == 'short' and df1m_h['low'].iloc[j] <= tp)):
                                winh = True
                                break
                            elif (side == 'long' and df1m_h['low'].iloc[j] <= sl) or (side == 'short' and df1m_h['high'].iloc[j] >= sl):
                                break
                            elif not break_even and ((side == 'long' and df1m_h['high'].iloc[j] >= be) or (side == 'short' and df1m_h['low'].iloc[j] <= be)):
                                if side == 'long': sl = round(op + tiny_profit*range_value, 1)
                                if side == 'short': sl = round(op - tiny_profit*range_value, 1)
                                break_even = True
                if not winh and ((side == 'long' and low[i] <= sl) or (side == 'short' and high[i] >= sl)):
                    if side == 'long': 
                        if print_log: print(f"Ticker: Price: {low[i]}")
                    if side == 'short': 
                        if print_log: print(f"Ticker: Price: {high[i]}")
                    if print_log and not break_even: print(f"Ticker: SL hit: {sl}")
                    if print_log and break_even: print(f"Ticker: SL (break-even) hit: {sl}")
                    usd_commission = size*(op_commission * op + sl_commission * sl)
                    if not break_even:
                        usd_loss += percentage * balance + usd_commission
                        balance -= percentage * balance + usd_commission
                        n_loss += 1
                    else:
                        usd_loss += usd_commission
                        balance += tiny_profit*percentage * balance-usd_commission
                        n_be += 1
                    if not break_even:val_result.append({'pred' : prediction_1, 'result' : 'loss', 'n_candles' : i - i_open, 'range': range_value, 'ny_open': trading_hours[i_open-1], 'ny_close': trading_hours[i], 'balance': balance}) 
                    if break_even:val_result.append({'pred' : prediction_1, 'result' : 'be', 'n_candles' : i - i_open, 'range': range_value, 'ny_open': trading_hours[i_open-1], 'ny_close': trading_hours[i], 'balance': balance})
                    if balance <= 200:
                        if print_log: print("Balance is too low, stopping trading.")
                        break
                else:
                    n_win += 1
                    if side == 'long': 
                        if print_log: print(f"Ticker: Price: {high[i]}")
                    if side == 'short': 
                        if print_log: print(f"Ticker: Price: {low[i]}")
                    if print_log: print(f"Ticker: TP hit: {tp}")
                    usd_commission = size*(op_commission * op + tp_commission * tp)
                    usd_win += ratio * percentage * balance - usd_commission
                    balance += ratio * percentage * balance - usd_commission
                    val_result.append({'pred' : prediction_1, 'result' : 'win', 'n_candles' : i - i_open, 'range': range_value, 'ny_open': trading_hours[i_open-1], 'ny_close': trading_hours[i], 'balance': balance}) 
                if print_log: 
                    print(f"\nTicker: Balance: {balance}\n\n")
                    pass
            elif (frac_be < 1 and not break_even and ((side == 'long' and high[i] > be) or (side == 'short' and low[i] < be))):
                if i==i_filled:
                    df1m_h = df1m[(df1m['date'] >= df_ori['date'].iloc[i]) & (df1m['date'] < df_ori['date'].iloc[i+1])].reset_index(drop=True)
                    filled_1m = False
                    for j in range(len(df1m_h)):
                        if not filled_1m and ((side == 'long' and df1m_h['low'].iloc[j] <= op) or (side == 'short' and df1m_h['high'].iloc[j] >= op)):
                            filled_1m = True
                        if filled_1m and ((side == 'long' and df1m_h['high'].iloc[j] >= be) or (side == 'short' and df1m_h['low'].iloc[j] <= be)):
                            break_even = True
                            break
                else:
                    break_even = True
                if break_even:
                    if side == 'long': sl = round(op + tiny_profit*range_value, 1) # Move SL to break-even
                    if side == 'short': sl = round(op - tiny_profit*range_value, 1) # Move SL to break-even
                    if print_log:
                        print(f"Ticker: Date: {df_ori['date_ny'].iloc[i]}")
                        print(f"Ticker: Price: {open_prices[i]}")
                        print(f"Ticker: Move SL to break-even: {sl}")
                        print("")
            elif (i - i_open >= 200) and ((side=='long' and open_prices[i] > op) or (side=='short' and open_prices[i] < op)):
                in_position = False
                n_market += 1
                usd_commission = size*(op_commission * op + sl_commission * open_prices[i])
                usd_loss += usd_commission
                usd_win += abs(open_prices[i]-op)*size-usd_commission
                balance += abs(open_prices[i]-op)*size-usd_commission
                val_result.append({'pred' : prediction_1, 'result' : 'market_close', 'n_candles' : i - i_open, 'range': range_value, 'ny_open': trading_hours[i_open-1], 'ny_close': trading_hours[i], 'balance': balance}) 
                if print_log:
                    print(f"Exiting position after {i - i_open} candles at market price {open_prices[i]}")
                    print(f"Ticker: Date: {df_ori['date_ny'].iloc[i]}")
                    print(f"Ticker: Price: {open_prices[i]}")
                    print(f"Ticker: Break-even exit")
                    print(f"\nTicker: Balance: {balance}\n\n")
    save_model = False
    pf = usd_win / usd_loss
    if do_month == 7: month='July'
    if do_month == 8: month='August'
    if do_month == 9: month='September'
    if do_month == 't': month='Training'
    message = ''
    if balance > 1070 and pf > 1.1:
        save_model = True
        message = f"Final {month} balance after trading: {int(balance)}, Wins: {n_win}, Losses: {n_loss}, BreakEven: {n_be}, MarketClose: {n_market}, usd_win: {int(usd_win)}, usd_loss: {int(usd_loss)}, Profit factor: {pf:.2f}"
    return balance, pf, usd_win, usd_loss, n_win, n_loss, n_be, n_market, save_model, message

frac_be, limit_low_range, limit_top_range, use_NY_trading_hour, use_day_month = 0.6, 130, 2000, True, 'None'  # Options: None, 'day', 'month
ma_periods, periods_look_back = np.arange(5,201,5), [10, 30, 50, 70, 90, 120, 140, 160, 180, 200]
path = '../test_12_all_models/test_12/'
for timeframe in ['15m','30m']:
    #if timeframe == '30m':continue
    models_tf = [f for f in os.listdir(path+f'models_{timeframe}') if f.endswith('.pt')]
    print(f'Found {len(models_tf)} models in models_{timeframe} folder\n')
    if len(models_tf) == 0: continue
    for model_name in models_tf:
        model_name = model_name.replace('.pt', '')

        timeframe, side, ratio, window, method, prices_col_1, prices_col_2, multiplier, period_lb, std_dev, shift_open = get_inputs(model_name)
        model = torch.load(path+f'models_{timeframe}/{model_name}.pt', weights_only=False, map_location=device)
        ct = joblib.load(path+f'models_{timeframe}/ct_{model_name}.pkl')

        message_all = ''
        for do_month in [10,9,8,7,'t']:
            balance, pf, usd_win, usd_loss, n_win, n_loss, n_be, n_market, save_model, message = run_forward_test(model, ct, timeframe, frac_be, limit_low_range, limit_top_range, shift_open, use_NY_trading_hour, use_day_month, method, prices_col_1, prices_col_2, period_lb, std_dev, multiplier, ratio, side, window, ma_periods, periods_look_back, cols_original=ct.feature_names_in_, X_train=None, do_month=do_month)
            if not save_model: break
            message_all += message+'\n'

        if save_model:
            print(model_name)
            print(f'Info: {timeframe} {side}, shift_open: {shift_open}, window: {window}, method: {method}, prices: {prices_col_1}-{prices_col_2} multiplier: {multiplier}, std_dev: {std_dev}, period_target: {period_lb}, ratio: {ratio}, use_NY_trading_hour: {use_NY_trading_hour}, use_day_month: {use_day_month}')
            print(message_all)
