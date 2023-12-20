import os
import torch
import torch.nn as nn
import pandas as pd
from dataset import *
calculate_loss_over_all_values = True
from matplotlib import pyplot
from para import input_window, output_window, plot_save_path


# entweder ist hier ein fehler im loss oder in der train methode, aber die ergebnisse sind unterschiedlich
# auch zu denen der predict_future
def evaluate(eval_model, data_seq, data_label, MSE_criterion, MAE_criterion):
    eval_model.eval()  # Turn on the evaluation mode
    MSE_total_loss = 0.
    MAE_total_loss = 0.
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_seq) - 1, eval_batch_size):
            data, targets = get_batch(data_seq, data_label, i, eval_batch_size)
            output = eval_model(data)
            print(output[-output_window:].size(), targets[-output_window:].size())
            if calculate_loss_over_all_values:
                MSE_total_loss += MSE_criterion(output, targets).item()
                MAE_total_loss += MAE_criterion(output, targets).item()
            else:
                MSE_total_loss += MSE_criterion(output[-output_window:], targets[-output_window:]).item()
                MAE_total_loss += MAE_criterion(output[-output_window:], targets[-output_window:]).item()

    return MSE_total_loss / len(data_seq), MAE_total_loss / len(data_seq)


def plot(eval_model, data_seq, data_label, epoch, scaler, MSE_criterion, MAE_criterion):
    # 输入数据格式：（序列窗口数， 每个窗口序列长度， 每条序列特征数）
    # （2000+， 96， 7）
    # （2000+， 336， 7）
    eval_model.eval()
    # 初始化损失
    MSE_total_loss = 0.
    MAE_total_loss = 0.
    # 初始化输出结果
    test_result = torch.Tensor(0)
    # 初始化真实值
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_seq) - 1):
            # 数据格式：（序列长度, 特征数）
            data, target = get_batch(data_seq, data_label, i, 1)
            # batch_size=1，故数据格式：（1, 序列长度, 特征数）
            data = data.unsqueeze(0)
            target = target.unsqueeze(0)
            # 递归预测： 0-96预测97-192 97-192预测193-288……
            # 第一个output输出是（1, 96, 7）
            output = eval_model(data)
            new_output = output.clone()
            if output_window > input_window:
                # 每次循环，output变成（1, 96*j, 7）
                for _ in range(0, output_window-input_window, input_window):
                    # # 如果j < len(data[0])，那么新的数据由预测数据和原数据组成
                    # if j < len(data[0]) - 1:
                    #     new_data = torch.cat((data[:, j + 1:, :], output[:, :j + 1, :]), 1)
                    # # 否则全由预测数据组成
                    # else:
                    #     new_data = output[:, j + 1 - len(data_label[0]):j + 1, :]
                    new_data = new_output
                    new_output = eval_model(new_data)
                    # output = torch.cat((output, new_output[:, -1, :].unsqueeze(1)), 1)
                    output = torch.cat((output, new_output), 1)
                output = output[:, :output_window, :]
            MSE_total_loss += MSE_criterion(output, target).item()
            MAE_total_loss += MAE_criterion(output, target).item()

            test_result = torch.cat((test_result, output[-1, :].squeeze(1).cpu()),
                                    0)
            truth = torch.cat((truth, target[-1, :].squeeze(1).cpu()), 0)

    test_result_ = scaler.inverse_transform(test_result[:300])
    truth_ = scaler.inverse_transform(truth)
    for m in range(7):
        test_result = test_result_[:, m]
        truth = truth_[:, m]
        fig = pyplot.figure(1, figsize=(20, 5))
        fig.patch.set_facecolor('xkcd:white')
        pyplot.plot([k + 96 for k in range(204)], test_result[96:], color="red")
        pyplot.title('Prediction uncertainty')
        pyplot.plot(truth[:300], color="black")
        pyplot.legend(["prediction", "true"], loc="upper left")
        ymin, ymax = pyplot.ylim()
        pyplot.vlines(96, ymin, ymax, color="blue", linestyles="dashed", linewidth=2)
        pyplot.ylim(ymin, ymax)
        pyplot.xlabel("Periods")
        pyplot.ylabel("Y")
        file_path = plot_save_path + str(epoch)
        if not os.path.exists(file_path):
            # 如果文件不存在，则创建文件
            os.makedirs(file_path)
        pyplot.savefig(file_path + '/' + str(m) + '.jpg')
        #  pyplot.show()
        pyplot.close()
    return MSE_total_loss / i, MAE_total_loss / i