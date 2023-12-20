import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot
import pandas as pd
from dataset import *
from model import *
from plot import *
from para import input_window, output_window, test_save_path, model_save_path

torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_ = pd.read_csv('data/ETTh1.csv', parse_dates=['date'])


calculate_loss_over_all_values = False
train_seq, train_label, test_seq, test_label, val_seq, val_label, scaler = get_data(data_, input_window, output_window)
model = TransAm().to(device)
MSE_criterion = nn.MSELoss()
MAE_criterion = nn.L1Loss()

model = torch.load(model_save_path)
epoch = 'test'
MSE_val_loss, MAE_val_loss = plot(model, test_seq, test_label, epoch, scaler, MSE_criterion, MAE_criterion)
print('test MSE loss: {:5.5f}| test MAE loss: {:5.5f}'.format(MSE_val_loss, MAE_val_loss))