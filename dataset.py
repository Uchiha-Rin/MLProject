import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
from para import input_window, output_window
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_inout_sequences(input_data, input_window, output_window):
    train_seq_list = []
    train_label_list = []
    L = len(input_data)
    for i in range(L - input_window - output_window):
        # train_seq = np.append(input_data[i:i+input_window,:][:-output_window,:] , np.zeros((output_window,7)),axis=0)
        # train_label = input_data[i:i+input_window,:]
        train_seq = input_data[i:i + input_window, :]
        train_label = input_data[i + input_window:i + input_window + output_window]
        train_seq_list.append(train_seq)
        train_label_list.append(train_label)
        # inout_seq.append((train_seq, train_label))
    return torch.FloatTensor(train_seq_list), torch.FloatTensor(train_label_list)


def get_data(data_, input_window, output_window):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = data_.loc[:, "HUFL":  "OT"]
    series = data.to_numpy()
    amplitude = scaler.fit_transform(series)

    train_sampels = round(len(amplitude) * 0.6)
    test_samples = round(len(amplitude) * 0.8)
    train_data = amplitude[:train_sampels]
    test_data = amplitude[train_sampels:test_samples]
    val_data = amplitude[test_samples:]

    train_seq, train_label = create_inout_sequences(train_data, input_window, output_window)

    # test_data = torch.FloatTensor(test_data).view(-1)
    test_seq, test_label = create_inout_sequences(test_data, input_window, output_window)

    val_seq, val_label = create_inout_sequences(val_data, input_window, output_window)

    return train_seq.to(device), train_label.to(device), test_seq.to(device), test_label.to(device), val_seq.to(device), val_label.to(device), scaler


def get_batch(seq, label, i, batch_size):
    seq_len = min(batch_size, len(seq) - 1 - i)
    # data = source[i:i + seq_len]
    # input = torch.stack(torch.stack([item for item in seq[i:i + seq_len]]).chunk(input_window, 1)).squeeze()  # 1 is feature size
    # target = torch.stack(torch.stack([item for item in label[i:i + seq_len]]).chunk(output_window, 1)).squeeze()
    input = torch.stack([item for item in seq[i:i + seq_len]]).squeeze()
    target = torch.stack([item for item in label[i:i + seq_len]]).squeeze()
    return input, target


