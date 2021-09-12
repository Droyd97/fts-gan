"""
Module for importing the data from a csv file into a usable Tensor for FTS-GAN.
"""
from re import L
from pandas.core import indexing
import torch
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
from datetime import datetime as dt

from ftsgan.preprocess import DiffTransform, LogTransform, StandardScaler, RollingWindow, Pipeline, Gaussianize, threshold


class FinancialSeries():

    def __init__(self, processed_series):
        self.series = processed_series

    def to_timeseries(self):
        series = []
        labels = {}
        for idx, (key, value) in enumerate(self.series.items()):
            series.append(value.data.unsqueeze(2))
            if value.labels is not None:
                labels[idx] = value.labels
        if labels:
            # return TimeSeries(torch.cat(series, dim=2), labels=torch.cat(labels, dim=1))
            return TimeSeries(torch.cat(series, dim=2), labels=labels)
        else:
            return TimeSeries(torch.cat(series, dim=2), None)


class Series():
    def __init__(self, data, raw_data, pipeline, labels=None):
        self.data = data.float()
        self.raw_data = raw_data.float()
        self.pipeline = pipeline
        self.labels = labels


class TimeSeries(Dataset):
    def __init__(self, data, labels=None):
        self.data = data.float()
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]
    
    def sample(self, batch_size):
        """
        Sample paths.
        :param batch_size: int.
        :return: batch
        """

        def get_indices(x, batch_size):
            return torch.multinomial(
                torch.arange(0, x.shape[0]).float(),
                num_samples=batch_size,
                replacement=False if x.shape[0] > batch_size else True
            )

        indices = get_indices(self.data, batch_size)
        if self.labels is None:
            return self.data[indices], None
        else:
            return self.data[indices], slice_dict_array(self.labels, indices)


def slice_dict_array(dict, indices):
    new_dict = {}
    for key, value in dict.items():
        new_dict[key] = value[indices]
    return new_dict


class ClassificationData():
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train.float()
        self.X_test = X_test.float()
        self.y_train = y_train.long()
        self.y_test = y_test.long()

    def sample(self, batch_size):

        def get_indices(x, batch_size):
            return torch.multinomial(
                torch.arange(0, x.shape[0]).float(),
                num_samples=batch_size,
                replacement=False if x.shape[0] > batch_size else True
            )

        indices = get_indices(self.X_train, batch_size)
        return self.X_train[indices], self.y_train[indices], self.X_test, self.y_test


def load_classification_data(
    real_data,
    window,
    lookahead,
    asset_condition,
    condition,
    train_split,
    synth_data=None,
    prices=True,
):
    data = np.array([])

    if synth_data is None:
        for asset, values in real_data.series.items():
            data = np.vstack([data, values.raw_data.squeeze().cpu().numpy()]) if data.size else \
            values.raw_data.squeeze().cpu().numpy()

        train_data = data[:, :int(train_split * data.shape[-1])]
        test_data = data[:, int((1 - train_split) * data.shape[-1]):]
        n_train = train_data.shape[-1]
        n_test = test_data.shape[-1]
    else:
        for asset, values in synth_data.items():
            data = np.vstack([data, values.squeeze().cpu().numpy()]) if data.size else \
            values.squeeze().cpu().numpy()

        train_data = data
        n_train = train_data.shape[-1]
        data = np.array([])
        for asset, values in real_data.series.items():
            data = np.vstack([data, values.raw_data.squeeze().cpu().numpy()]) if data.size else \
            values.raw_data.squeeze().cpu().numpy()
        test_data = data
        n_test = test_data.shape[-1]


    train_X = np.dstack([train_data[:, window * k: window + window * k] for k in range(0, int((n_train / window)))])
    train_X = np.swapaxes(train_X, 0, 2)
    train_X = np.swapaxes(train_X, 1, 2)
    print(train_X.shape)
    train_y = np.zeros(train_X.shape[0])

    if prices:
        for i in range(train_X.shape[0] - 1):
            train_y[i] = condition.process([train_data[asset_condition, window * i + window + lookahead]])
    else:
        for i in range(train_X.shape[0] - 1):
            train_y[i] = condition.process([train_data[asset_condition, window * i + window + lookahead]], [train_data[asset_condition, window * i + window]])

    test_X = np.dstack([test_data[:, k: window + k] for k in range(0, int((n_test - window - lookahead)))])
    test_X = np.swapaxes(test_X, 0, 2)
    test_X = np.swapaxes(test_X, 1, 2)

    test_y = np.zeros(test_X.shape[0])

    if prices:
        for i in range(test_X.shape[0]):
            test_y[i] = condition.process([test_data[asset_condition, i + window + lookahead]])
    else:
        for i in range(test_X.shape[0]):
            test_y[i] = condition.process([test_data[asset_condition, i + window + lookahead]], [test_data[asset_condition, i + window]])

    dataset = ClassificationData(torch.tensor(train_X, device=values.raw_data.device),
                                 torch.tensor(train_y, device=values.raw_data.device),
                                 torch.tensor(test_X, device=values.raw_data.device),
                                 torch.tensor(test_y, device=values.raw_data.device))

    return dataset
        
        
class PriceThreshold():

    def __init__(self, thresholds):
        self.thresholds = thresholds

    def process(self, x, current_price=None):
        if current_price is None:
            return threshold(x, self.thresholds)
        else:
            returns = np.log(np.array(x) / np.array(current_price)) * 100
            return threshold(returns, self.thresholds)

# class ReturnThreshold():
#     def __init__(self, thresholds):
#         self.thresholds = thresholds

#     def process(self, x):


# TODO: Refactor
def load_data(
    file_dir,
    columns,
    window_size,
    stride=1,
    start_date='2000-01-01',
    end_date=None,
    calendar='NYSE',
    gaussianize=False,
    format='%d/%m/%Y',
    device=torch.device('cpu'),
    diff_log=None,
):
    # Get data and convert to Tensor
    prices = pd.read_csv(file_dir)
    prices['Date'] = pd.to_datetime(prices['Date'], format=format)
    prices.set_index('Date', inplace=True)

    if end_date is None:
        end_date = dt.today().strftime('%Y-%m-%d')

    if dt.strptime(start_date, '%Y-%m-%d') < prices.index[0]:
        start_date = prices.index[0]

    if dt.strptime(end_date, '%Y-%m-%d') > prices.index[-1]:
        end_date = prices.index[-1]

    if calendar is not None:
        cal = mcal.get_calendar(calendar)
        valid = cal.valid_days(start_date=start_date, end_date=end_date)

        valid = valid.strftime('%Y-%m-%d').to_list()
        prices = prices.loc[valid]
    else:
        prices = prices[start_date : end_date]

    data = torch.tensor(prices[columns].to_numpy(), device=device)
    data = data.transpose(0, 1)
    processed_data = {}
    for idx, column in enumerate(columns):
        processed_column = data[idx].unsqueeze(0)
        raw_data = processed_column
        log_prices = LogTransform()
        log_returns = DiffTransform()
        scale1 = StandardScaler()

        window = RollingWindow(window_size, stride)
        steps = [log_prices]
        if diff_log is None:
            steps += [log_returns]
        elif diff_log[column]:
            steps += [log_returns]
        steps += [scale1]
        if gaussianize:
            gauss = Gaussianize()
            scale2 = StandardScaler()
            steps += [gauss]
            steps += [scale2]
        pipe = Pipeline(steps)
        processed_column = pipe.transform(processed_column)
        if diff_log is None:
            zeros = torch.zeros(processed_column.shape[0], 1, device=device)
            processed_column = torch.cat((zeros, processed_column), dim=-1)
        elif diff_log[column]:
            zeros = torch.zeros(processed_column.shape[0], 1, device=device)
            processed_column = torch.cat((zeros, processed_column), dim=-1)
        processed_column = window.transform(processed_column)
        processed_data[column] = Series(processed_column, raw_data, pipe)

    dataset = FinancialSeries(processed_data)
    return dataset


def load_multi_data(
    file_dir,
    assets,
    variables,
    window_size,
    stride=1,
    start_date='2000-01-01',
    end_date=None,
    calendar='NYSE',
    gaussianize=False,
    format='%d/%m/%Y',
    device=torch.device('cpu'),
    diff_log=None,
    condition=None,
    threshold=None,
    asset_apply=None,
):
    # Get data and convert to Tensor
    processed_data = {}
    if asset_apply is None:
        asset_apply = [True] * len(assets)
    for idx, asset in enumerate(assets):
        prices = pd.read_csv(file_dir[idx])
        prices['Date'] = pd.to_datetime(prices['Date'], format=format)
        prices.set_index('Date', inplace=True)

        if end_date is None:
            end_date = dt.today().strftime('%Y-%m-%d')

        if dt.strptime(start_date, '%Y-%m-%d') < prices.index[0]:
            start_date = prices.index[0]

        if dt.strptime(end_date, '%Y-%m-%d') > prices.index[-1]:
            end_date = prices.index[-1]

        cal = mcal.get_calendar(calendar)
        valid = cal.valid_days(start_date=start_date, end_date=end_date)
        valid = valid.strftime('%Y-%m-%d').to_list()
        prices = prices.loc[valid]

        # prices = prices.loc[start:]
        data = torch.tensor(prices[variables[asset]].to_numpy(), device=device)
        data = data.transpose(0, 1)
        raw_data = data

        log_prices = LogTransform()
        log_returns = DiffTransform()
        scale1 = StandardScaler()

        window = RollingWindow(window_size, stride)
        steps = [log_prices]
        if diff_log is None:
            steps += [log_returns]
        elif diff_log[asset]:
            steps += [log_returns]
        steps += [scale1]
        if gaussianize:
            gauss = Gaussianize()
            scale2 = StandardScaler()
            steps += [gauss]
            steps += [scale2]
        pipe = Pipeline(steps)
        data = pipe.transform(data)
        if diff_log is None:
            zeros = torch.zeros(data.shape[0], 1, device=device)
            data = torch.cat((zeros, data), dim=-1)
        elif diff_log[asset]:
            zeros = torch.zeros(data.shape[0], 1, device=device)
            data = torch.cat((zeros, data), dim=-1)
        data = window.transform(data)
        labels = None
        if condition is not None and threshold is not None:
            if asset_apply[idx]:
                cond = condition(threshold)
                labels = cond.fit(data)
        processed_data[asset] = Series(data, raw_data, pipe, labels)

    return FinancialSeries(processed_data)
