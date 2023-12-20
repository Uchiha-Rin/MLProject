import torch
import torch.nn as nn
import numpy as np
import time
import math
from matplotlib import pyplot
import pandas as pd
import os
from dataset import *
from model import *
from plot import *
from para import input_window, output_window, model_save_path, loss_save_path, batch_size, epochs, lr


def train(train_seq, train_label, criterion):
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_seq) - 1, batch_size)):
        data, targets = get_batch(train_seq, train_label, i, batch_size)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_seq) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_seq) // batch_size, scheduler.get_lr()[0],
                              elapsed * 1000 / log_interval,
                cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()



torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_ = pd.read_csv('data/ETTh1.csv', parse_dates=['date'])


# 获得数据格式：（序列窗口数， 每个窗口序列长度， 每条序列特征数）
train_seq, train_label, test_seq, test_label, val_seq, val_label, scaler = get_data(data_, input_window, output_window)

# 建立模型
model = TransAm().to(device)
# 建立损失函数
MSE_criterion = nn.MSELoss()
MAE_criterion = nn.L1Loss()
# 选取SGD和Adam作为优化函数
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)

# 初始化最佳模型
best_model = None

# 初始化最佳损失
last_val_loss = float("inf")

# 保存损失
MSE_loss_list = []
MAE_loss_list = []

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    # 使用训练集进行训练，输入和输出的序列长度应该一致
    train(train_seq, train_label[:, :input_window, :], MSE_criterion)
    # 使用验证集验证
    if (epoch % 1 == 0):
        MSE_val_loss, MAE_val_loss = plot(model, val_seq, val_label, epoch, scaler, MSE_criterion, MAE_criterion)
    else:
        MSE_val_loss, MAE_val_loss = evaluate(model, val_seq, val_label, MSE_criterion, MAE_criterion)
    MSE_loss_list.append(MSE_val_loss)
    MAE_loss_list.append(MAE_val_loss)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (
                time.time() - epoch_start_time),
                                                                                                  MSE_val_loss,
                                                                                                  math.exp(MSE_val_loss)))
    print('-' * 89)
    # 如果loss增加，则停止
    if MSE_val_loss > last_val_loss:
        torch.save(best_model, model_save_path)
        break
    else:
        last_val_loss = MSE_val_loss
        best_model = model

    scheduler.step()

# 保存loss
fig = pyplot.figure(1, figsize=(20, 5))
fig.patch.set_facecolor('xkcd:white')
pyplot.title('loss')
pyplot.plot(MSE_loss_list, color="black")
pyplot.plot(MAE_loss_list, color="red")
pyplot.legend(["MSE loss", "MAE loss"], loc="upper left")
pyplot.xlabel("epoch")
pyplot.ylabel("loss value")
file_path = loss_save_path
if not os.path.exists(file_path):
    # 如果文件不存在，则创建文件
    os.makedirs(file_path)
pyplot.savefig(file_path + '/loss.jpg')
#  pyplot.show()
pyplot.close()

# 计算std
MSE_std = np.std(MSE_loss_list, ddof=1)
MAE_std = np.std(MAE_loss_list, ddof=1)
# 保存std
with open(file_path + '/std.txt', 'w', encoding='utf-8') as f:
    f.write('MSE_std:' + str(MSE_std) + '\t' + 'MAE_std' + str(MAE_std))

if not os.path.exists(model_save_path):
    torch.save(model, model_save_path)


