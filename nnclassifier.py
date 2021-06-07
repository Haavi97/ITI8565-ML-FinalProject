import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score

device = 'cpu'


class trainData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class testData(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


class binaryClassification(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(binaryClassification, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_out = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim)
        self.batchnorm2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.dropout(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


def train(model, epochs, train_loader, lrr=0.0001):
    device = 'cpu'
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lrr)
    model.train()
    result1 = []
    result2 = []
    for e in range(1, epochs+1):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))
            #f1 = f1_score(y_pred, y_batch.unsqueeze(1))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        acc = epoch_acc/len(train_loader)
        loss_ = epoch_loss/len(train_loader)

        result1.append(acc,)
        result2.append(loss_)

        print(
            f'Epoch {e+0:03}: | Loss: {loss_:.5f} | Acc: {acc:.3f}')
    return result1, result2


def evaluate(model, test_loader, y_test):
    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())

    y_pred_list = [e.squeeze().tolist() for e in y_pred_list]
    # y_test = [round(e) for e in y_test]

    print('Confusion matrix:')
    print(confusion_matrix(y_test, y_pred_list))

    print(classification_report(y_test, y_pred_list))


def nn_do(X, y, input_dim, epochs=50, lr=0.0001):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=69)

    EPOCHS = epochs
    BATCH_SIZE = 64
    LEARNING_RATE = lr

    train_data = trainData(torch.FloatTensor(X_train),
                           torch.FloatTensor(y_train))
    test_data = testData(torch.FloatTensor(X_test))
    train_loader = DataLoader(
        dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=1)

    model = binaryClassification(input_dim, 64, dropout=0.2)
    model.to(device)
    print(model)
    result = train(model, EPOCHS, train_loader, lrr=LEARNING_RATE)
    evaluate(model, test_loader, y_test)
    return result
