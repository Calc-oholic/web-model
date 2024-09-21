# model.py
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from datetime import date
import talib as ta
import warnings

# Set up warnings
warnings.filterwarnings('ignore')

todayString = date.today().strftime('%Y-%m-%d')

def get_relative_stock_data(ticker, start, end=todayString, interval="1d"):
    data = yf.download(ticker, start=start, end=end, interval=interval)
    rsi = ta.RSI(data['Close'], timeperiod=14)
    data['RSI'] = rsi

    data_rel_to_prev = data.pct_change().dropna().reset_index().rename(columns={"index": "Date"})
    data = data.dropna().reset_index().rename(columns={"index": "Date"})

    return data, data_rel_to_prev


def scroll_data(original_data, data, context_window=10, solution_window=3):
    original_columns = data.columns
    scrolled_data_rel = pd.DataFrame()

    for i in range(solution_window):
        scrolled_data_rel[f"Solution{'Close'}{i + 1}"] = data["Close"].shift(-i)

    # Context data preparation
    for col in data.columns:
        if col != "Date" and not "Solution" in col:
            for i in range(context_window):
                scrolled_data_rel[f"ContextRelPrev{col}{-i}"] = data[col].shift(i)

    scrolled_data_abs = pd.DataFrame()
    # Absolute data preparation
    for col in original_data.columns:
        if col != "Date" and not "Solution" in col and not "Context" in col:
            for i in range(context_window):
                scrolled_data_abs[f"ContextRelStart{col}{-i}"] = original_data[col].shift(i)

            for i in range(context_window):
                scrolled_data_abs[f"ContextRelStart{col}{-i}"] = scrolled_data_abs[f"ContextRelStart{col}{-i}"] / \
                                                                 scrolled_data_abs[
                                                                     f"ContextRelStart{col}{-context_window + 1}"] - 1

    scrolled_data = pd.concat([scrolled_data_rel, scrolled_data_abs], axis=1).dropna().reset_index(drop=True)
    return scrolled_data


def unrelativize_data_array(data):
    processed_data = data.copy()
    if len(processed_data) == 1:
        processed_data = processed_data[0]

    processed_data[0] = 0
    for i in range(len(processed_data)):
        if i != 0:
            processed_data[i] = (processed_data[i] + 1) * (processed_data[i - 1] + 1) - 1

    return processed_data


def train_model(ticker, start_date, context_window, solution_window):
    original_data, relative_data = get_relative_stock_data(ticker, start_date)

    data = scroll_data(original_data, relative_data, context_window=context_window, solution_window=solution_window)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    # Split data
    train, test = train_test_split(data, test_size=0.1)

    contextColumns = data.columns[solution_window:]
    solutionColumns = data.columns[:solution_window]

    X_train = train[contextColumns]
    y_train = train[solutionColumns]

    X_test = test[contextColumns]
    y_test = test[solutionColumns]

    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)

    # actual tolerance!
    tolerance = 0.000000001
    # tolerance = 0.001

    model = MLPRegressor(hidden_layer_sizes=(250, 50, 250, 10), activation='relu', solver='adam',
                         alpha=0.0001, learning_rate='adaptive', learning_rate_init=0.0005,
                         verbose=True, tol=tolerance, max_iter=500)

    # TRAIN MODEL
    model.fit(X_train, y_train)

    # predict last 100
    where_to_predict = 100
    context_to_predict = data[contextColumns][-1 - where_to_predict:-where_to_predict]
    solution_to_predict = data[solutionColumns][-1 - where_to_predict:-where_to_predict]

    predictions = model.predict(context_to_predict)

    mse = mean_squared_error(y_test, model.predict(X_test))

    plotted_context = context_to_predict.iloc[:, 3*context_window:4*context_window].values[0][::-1].reshape(-1, 1)
    plotted_solution = solution_to_predict.values.reshape(-1, 1)

    return predictions, mse, plotted_context, plotted_solution