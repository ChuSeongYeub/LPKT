# coding: utf-8
# started on 2022/3/22 @zelo2
# finished on 2022/4/1 @zelo2

from sklearn.model_selection import train_test_split
import torch
import numpy as np
import tqdm
import pandas as pd
import option_LPKTNet_copy
import LPKT_dataloader
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import sklearn.metrics as metrics
from tqdm import tqdm
device = torch.device('cuda:0')


def train():
    data_inf = pd.read_pickle('ednet/lpkt_ednet_num_list.pkl')
    raw_data = pd.read_pickle('ednet/lpkt_ednet_data.pkl')
    q_matrix = pd.read_pickle('ednet/lpkt_ednet_qmatrix.pkl').to(device)

    '''Paramater Initialization'''
    stu_num = data_inf[0]
    exercise_num = data_inf[1]
    skill_num = data_inf[2]
    answer_time_num = data_inf[3]
    interval_time_num = data_inf[4]
    option_num = data_inf[5]

    batch_size = 90
    train_ratio = 0.8
    learning_rate = 0.001
    optimizer = 'adam'
    num_epochs = 20

    train = []
    test = []

    train_size = int(len(raw_data) * train_ratio)
    test_size = len(raw_data) - train_size

    train_dataset, test_dataset = train_test_split(
        raw_data, test_size=test_size, random_state=42)

    train_dataset = LPKT_dataloader.lpkt_dataset(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = LPKT_dataloader.lpkt_dataset(test_dataset)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True)

    '''Inilization'''
    net = option_LPKTNet_copy.LPKTNet(exercise_num, skill_num, stu_num, answer_time_num, interval_time_num, option_num,
                                      d_k=128, d_a=50, d_e=128, q_matrix=q_matrix)
    net = net.to(device)
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)

    acc = []
    auc = []
    loss_means = []

    '''Train and Validation'''
    for epoch in range(num_epochs+1):
        train_loss = []
        print('Epoch', epoch + 1)
        '''Train'''
        for data in tqdm(train_dataloader):
            input_data = data[0]
            labels = data[1]
            optimizer.zero_grad()
            labels = labels.float().to(device)
            input_data = input_data.to(device)
            exercise_id = input_data[:, 0].long()  # [batch_size, sequence]
            answer_time = input_data[:, 1].long()
            interval_time = input_data[:, 2].long()
            option_value = input_data[:, 3].long()
            answer_value = input_data[:, 4].float()

            net.train()

            pred = net(exercise_id, answer_time,
                       interval_time, option_value, answer_value)
            pred = pred[:, 1:].to(device)

            '''Backward propagation'''
            loss = loss_function(pred, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            loss_mean = np.mean(train_loss)

            # running_loss += loss.item()
        # print('Loss:', running_loss)

        '''Test'''
        test_pred = []
        test_labels = []
        with torch.no_grad():
            for data in tqdm(test_dataloader):
                input_data = data[0]
                labels = data[1]
                labels = labels[:, -1].float().to(device)
                input_data = input_data.to(device)
                exercise_id = input_data[:, 0].long()  # [batch_size, sequence]
                answer_time = input_data[:, 1].long()
                interval_time = input_data[:, 2].long()
                option_value = input_data[:, 3].long()
                answer_value = input_data[:, 4].float()

                net.eval()

                pred = net(exercise_id, answer_time,
                           interval_time, option_value, answer_value)
                pred = pred[:, -1].to(device)

                test_pred.append(pred)
                test_labels.append(labels)

            test_pred = torch.cat(test_pred).cpu().numpy()
            test_labels = torch.cat(test_labels).cpu().numpy()

            '''AUC'''
            test_auc = metrics.roc_auc_score(test_labels, test_pred)

            '''ACC'''
            test_pred[test_pred >= 0.5] = 1
            test_pred[test_pred < 0.5] = 0
            test_acc = np.nanmean((test_pred == test_labels) * 1)

            acc.append(test_acc)
            auc.append(test_auc)

            print(
                "Epoch: {},   ACC: {: .4f}, AUC: {: .4f},   Loss Mean: {: .4f}"
                .format(epoch, test_acc, test_auc, loss_mean)
            )

            # print("Test Results (AUC, Acc)", epoch, ":", test_auc, test_acc)

        # print("Final Results (AUC, Acc):", np.mean(
        #    np.array(auc)), np.mean(np.array(acc)))


if __name__ == '__main__':
    train()
