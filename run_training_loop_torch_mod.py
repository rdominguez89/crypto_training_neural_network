from sklearn.compose import ColumnTransformer
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from sklearn.metrics import accuracy_score#, precision_score, recall_score, roc_auc_score
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import random
import copy
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

# PyTorch EarlyStopping equivalent
class ImprovedDualEarlyStopping:
    def __init__(self, best_val_acc, best_train_acc, patience=10, min_epochs=30):
        self.patience = patience
        self.min_epochs = min_epochs
        self.best_epoch = 0
        self.best_val_acc = best_val_acc
        self.best_train_acc = best_train_acc
        self.best_val_loss = float('inf')
        self.best_state_dict = None
        self.wait = 0

    def check(self, epoch, train_acc, val_acc, val_loss, model):
        if ((epoch > 5000 and train_acc < 0.70) or (epoch > 9000 and train_acc < 0.74)) and self.best_epoch == 0:
            if epoch > 90:
                print(f"\nEpoch {epoch+1}: binary_accuracy stayed below 0.74. Stopping training.\n")
            else:
                print(f"\nEpoch {epoch+1}: binary_accuracy stayed below 0.70. Stopping training.\n")
            return True
        if (val_acc > self.best_val_acc and train_acc > self.best_train_acc):
            print(f"Epoch {epoch+1}: binary_accuracy improved from {self.best_train_acc:.4f} to {train_acc:.4f} or val_binary_accuracy improved from {self.best_val_acc:.4f} to {val_acc:.4f}.")
            self.best_train_acc = train_acc
            self.best_val_acc = val_acc
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.best_state_dict = {
                key: value.clone().detach() 
                for key, value in model.state_dict().items()
            }
            self.wait = 0
        elif self.best_epoch != 0:
            self.wait += 1
            if self.wait >= self.patience and epoch >= self.min_epochs:
                #model.load_state_dict(self.best_state_dict)
                print(f"Early stopping at epoch {self.best_epoch+1}/{epoch+1}. Best acc: {self.best_train_acc:.4f} Best val_acc: {self.best_val_acc:.4f}")
                return True
        return False

    def get_best(self):
        return self.best_epoch, self.best_val_acc, self.best_train_acc, self.best_state_dict


def prepare_data(df, side, ratio, remove_columns):
    remove_columns += [f'{side}_entry']
    X = df.drop(remove_columns, axis=1)
    cols_original = X.columns.tolist()
    y = df[f'{side}_entry']

    columns_to_scale = [col for col in X.columns if ((df[col] < 0) | (df[col] > 1)).any()]
    columns_to_leave = [col for col in X.columns if col not in columns_to_scale]

    ct = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), columns_to_scale),
            ('passthrough', 'passthrough', columns_to_leave)
        ]
    )
    X = ct.fit_transform(X)

    # Ensure X is a dense numpy array
    from scipy.sparse import spmatrix
    if isinstance(X, spmatrix):
        X = np.array(X.todense())
    elif not isinstance(X, np.ndarray):
        X = np.array(X)

    # Ensure y is a numpy array
    if hasattr(y, 'to_numpy'):
        y = y.to_numpy()
    y = np.array(y).flatten()

    split_idx_train = int(0.7 * X.shape[0])
    split_idx_val = int(0.85 * X.shape[0])

    X_train, X_val, X_test = X[:split_idx_train], X[split_idx_train:split_idx_val], X[split_idx_val:]
    y_train, y_val, y_test = y[:split_idx_train], y[split_idx_train:split_idx_val], y[split_idx_val:]

    # Use np.unique for class counts
    orig_counts = dict(zip(*np.unique(y, return_counts=True)))
    train_counts = dict(zip(*np.unique(y_train, return_counts=True)))
    val_counts = dict(zip(*np.unique(y_val, return_counts=True)))
    test_counts = dict(zip(*np.unique(y_test, return_counts=True)))

    if min(train_counts.values()) == 0 or max(train_counts.values())/min(train_counts.values()) > ratio + 0.5:
        pass
        #return None, None, None, None, None, None, None, None

    print(f'\nside: {side} Original: {orig_counts} Train: {train_counts} Val: {val_counts} Test: {test_counts}')

    # Apply over/under sampling only to the training data
    ros = RandomOverSampler(random_state=42)
    X_train, y_train = ros.fit_resample(X_train, y_train)

    # # Data augmentation: noise injection
    # def augment_data(X, y, noise_std=0.01):
    #     X_aug = X + np.random.normal(0, noise_std, X.shape)
    #     return np.vstack([X, X_aug]), np.hstack([y, y])

    # X_train, y_train = augment_data(X_train, y_train)

    return X_train, y_train, X_val, y_val, X_test, y_test, ct, cols_original


def create_model(X_train, num_layers, num_neurons_list, dropout_rate):
    seed = 52
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    layers = []
    input_dim = X_train.shape[1]

    # Input layer
    layers.append(nn.Linear(input_dim, num_neurons_list[0]))
    #layers.append(nn.BatchNorm1d(num_neurons_list[0]))
    layers.append(nn.SELU())
    layers.append(nn.Dropout(dropout_rate))

    # Hidden layers
    for i in range(1, num_layers):
        layers.append(nn.Linear(num_neurons_list[i-1], num_neurons_list[i]))
        #layers.append(nn.BatchNorm1d(num_neurons_list[i]))
        layers.append(nn.SELU())
        current_dropout = max(0, dropout_rate - (i * 0.05))
        layers.append(nn.Dropout(current_dropout))

    # Output layer
    layers.append(nn.Linear(num_neurons_list[-1], 1))
    layers.append(nn.Sigmoid())

    model = nn.Sequential(*layers)
    return model



def run_neural(df, side, ratio, remove_columns, timeframe, frac, limit_low_range, limit_top_range, shift_open, use_NY_trading_hour, use_day_month, method, prices_col_1, prices_col_2, period_target, std_dev, multiplier, window, ma_periods, periods_look_back):
    # Prepare the data
    X_train, y_train, X_val, y_val, X_test, y_test, ct, cols_original = prepare_data(df, side, ratio, remove_columns)
    if X_train is None: return
    acc_0 = 0.6
    device = 'cuda'
    neurons_list = [0]
    best_val_acc = acc_0
    best_train_acc = acc_0
    best_test_acc = acc_0
    best_neurons = 0
    print_log = True
    dropout_rate = 0.5
    while True:
        better_conf_found = False
        for neurons in range(32, 257, 32):
            if len(neurons_list)>1 and neurons > neurons_list[-2]: break
            #if neurons > 64 or len(neurons_list)>1:continue #borrar
            try:
                del model
                del optimizer
                torch.cuda.empty_cache()
            except:
                pass
            neurons_list[-1] = int(neurons)
            num_layers = len(neurons_list)
            model = create_model(X_train, num_layers, neurons_list, dropout_rate).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
            criterion = nn.BCELoss()
            early_stop = ImprovedDualEarlyStopping(best_val_acc, best_train_acc, patience=30, min_epochs=100)

            if y_train is None:
                return

            X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
            y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device).view(-1, 1)
            X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
            y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device).view(-1, 1)
            X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
            y_test_t = torch.tensor(y_test, dtype=torch.float32, device=device).view(-1, 1)
            if print_log:
                print(f"\nTraining model with configuration: {neurons_list} layers, neurons per layer: {neurons_list}.")
            for epoch in range(1000):
                model.train()
                optimizer.zero_grad()
                y_pred = model(X_train_t)
                loss = criterion(y_pred, y_train_t)
                loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val_t)
                    val_loss = criterion(val_pred, y_val_t).item()
                    y_pred = model(X_train_t)
                    train_acc = accuracy_score(y_train_t.cpu(), (y_pred.cpu() > 0.5).float())
                    val_acc = accuracy_score(y_val_t.cpu(), (val_pred.cpu() > 0.5).float())
                    if print_log and ((epoch + 1) % 150000 == 0 or epoch == -10):
                        print(f"Epoch {epoch+1}: Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
                #scheduler.step(val_loss)
                early_stop.check(epoch, train_acc, val_acc, val_loss, model)
                if early_stop.wait >= early_stop.patience and epoch >= early_stop.min_epochs:
                    break

            best_epoch_h, best_val_acc_h, best_train_acc_h, best_model_state_dict_h = early_stop.get_best()
            if best_epoch_h == 0:
                continue
            if best_val_acc_h > best_val_acc:
                model.load_state_dict(best_model_state_dict_h)
                model.eval()
                with torch.no_grad():
                    test_pred = model(X_test_t)
                    test_acc = accuracy_score(y_test_t.cpu(), (test_pred.cpu() > 0.5).float())
                if test_acc < best_test_acc: continue
                best_test_acc =  test_acc
                best_epoch = best_epoch_h
                better_conf_found = True
                best_train_acc = best_train_acc_h
                best_val_acc = best_val_acc_h
                best_neurons = neurons_list.copy()
                best_model = copy.deepcopy(model)   # Save best model for later
                print(f"New best neurons configuration: {best_neurons} best_epoch: {best_epoch}")
                print(f"Training Binary Accuracy: {best_train_acc:.4f}")
                print(f"Val Binary Accuracy: {best_val_acc:.4f}")
                print(f"Test Binary Accuracy: {test_acc:.4f}\n")
                pass
                # better_conf_found = False
                #if neurons==64:break
        if not better_conf_found and best_neurons != 0:
            from run_forward import run_forward_test
            print(f'\nInfo: {timeframe} {side}, shift_open: {shift_open}, window: {window}, method: {method}, prices: {prices_col_1}-{prices_col_2} multiplier: {multiplier}, std_dev: {std_dev}, period_target: {period_target}, ratio: {ratio}, use_NY_trading_hour: {use_NY_trading_hour}, use_day_month: {use_day_month}')
            print(f"Best neurons configuration: {best_neurons} with Train Acc: {best_train_acc:.4f} Val Acc: {best_val_acc:.4f} Test Acc: {best_test_acc:.4f}")
            # Recreate model and load best weights before forward test
            # num_layers = len(best_neurons)
            # best_model = create_model(X_train, num_layers, best_neurons, dropout_rate).to(device)
            # best_model.load_state_dict(best_model_state_dict)
            best_model.eval()
            balance_july, pf_july, usd_win, usd_loss, n_win, n_loss, n_be, n_market, save_model_july = run_forward_test(best_model, ct, timeframe, frac, limit_low_range, limit_top_range, shift_open, use_NY_trading_hour, use_day_month, method, prices_col_1, prices_col_2, period_target, std_dev, multiplier, ratio, side, window, ma_periods, periods_look_back, cols_original, X_train, do_month=7)
            if save_model_july:
                balance_august, pf_august, usd_win, usd_loss, n_win, n_loss, n_be, n_market, save_model_august = run_forward_test(best_model, ct, timeframe, frac, limit_low_range, limit_top_range, shift_open, use_NY_trading_hour, use_day_month, method, prices_col_1, prices_col_2, period_target, std_dev, multiplier, ratio, side, window, ma_periods, periods_look_back, cols_original, X_train, do_month=8)
                if save_model_august:
                    file_to_save = f'best_model_{timeframe}_{side}_ratio_{ratio}_window_{window}_method_{method}_{prices_col_1}_{prices_col_2}_mx_{multiplier}_period_{period_target}_std_{std_dev}_shift_{shift_open:.1f}_balj_{int(balance_july)}_pfj_{pf_july:.2f}_bala_{int(balance_august)}_pfa_{pf_august:.2f}'
                    # torch.save(best_model, f'models_{timeframe}/{file_to_save}.pt')
                    # joblib.dump(ct, f'models_{timeframe}/ct_{file_to_save}.pkl')
                    print(f"Model saved as {file_to_save}.pt")
            print('\n')   
            break
        elif best_neurons == 0:
            print("\nNo good configuration found, stopping search.\n")
            break
        neurons_list = list(best_neurons) + [0]
        print(f"Expanding to {len(neurons_list)} layers with initial neurons: {neurons_list}")





# new: print epoch summary every `period` epochs
# Remove EpochPrinter, handled in training loop



