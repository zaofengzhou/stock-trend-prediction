import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import quandl
import numpy as np
from datetime import date
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, (h, c) = self.lstm(x, (h0, c0))
        out = self.fc(h.squeeze(0))
        return out


class StockDataset(Dataset):
    def __init__(self, file_path, time_stamp, mode="train"):
        # with open(file_path, "r") as f:
        #     data = list(csv.reader(f))
        #     data = np.array(data[1:])[:, 1:].astype(float)
        self.data = pd.read_csv(file_path, engine="python")
        self.data = self.data.dropna()
        # self.data = self.data.drop(self.data[(self.data == 0).any(axis=1)].index)
        self.data = self.data[['Open', 'High', 'Low', 'Close', 'Volume']]

        labels = [-1]
        previous_close = None
        for index, row in self.data.iterrows():
            if previous_close is not None:
                if row['Close'] > previous_close:
                    labels.append(1)
                else:
                    labels.append(0)
            previous_close = row['Close']

        self.labels = np.where(self.data['Close'].shift(-1) > self.data['Close'], 1, 0)

        # 归一化
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(self.data)

        self.x = []
        self.y = []

        for i in range(time_stamp, len(scaled_data)):
            self.x.append(scaled_data[i - time_stamp:i])
            self.y.append(self.labels[i])
        self.x = torch.FloatTensor(self.x)
        self.y = torch.LongTensor(self.y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


if __name__ == '__main__':
    ss_index = pd.read_csv("/Users/zfzhou/Downloads/000001.SS.csv", engine="python")
    print(ss_index.shape)
    # ss_index.tail()
    ss_index.head()

    # missing_values_rows = ss_index[ss_index.isnull().any(axis=1) | (ss_index == 0).any(axis=1)]
    missing_values_rows = ss_index[ss_index.isnull().any(axis=1)]

    ss_index = ss_index.drop(missing_values_rows.index)

    ss_index['Date'] = pd.to_datetime(ss_index['Date'])
    ss_index.set_index('Date', inplace=True)

    plt.figure(figsize=(16, 8))
    # plt.figure(figsize=(14, 7))
    plt.plot(ss_index['Close'])
    plt.show()

    # 时间点长度
    time_stamp = 7

    epochs = 30
    batch_size = 64

    stock_dataset = StockDataset("/Users/zfzhou/Downloads/000001.SS.csv", time_stamp)
    print(stock_dataset.__len__())
    print(stock_dataset.__getitem__(0))

    # 划分训练集与验证集
    total_size = len(stock_dataset)
    train_size = int(total_size * 0.7)
    valid_size = total_size - train_size

    indices = list(range(total_size))
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]

    train_dataset = torch.utils.data.Subset(stock_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(stock_dataset, valid_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    input_size = stock_dataset.data.shape[-1]
    hidden_size = 64
    num_layers = 1
    model = LSTM(input_size, hidden_size, num_layers)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss = train_loss / len(train_loader.dataset)
        # print(f"Epoch {epoch + 1}, Train Loss: {train_loss}")
        print("Epoch : {}, Train Loss : {}".format(epoch + 1, train_loss))

        model.eval()  # 声明
        correct = 0.0  # 计算正确率
        test_loss = 0.0  # 测试损失
        acc = 0.0  # 准确率

        with torch.no_grad():
            for inputs, labels in valid_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

            test_loss = test_loss / len(valid_loader.dataset)
            accuracy = correct / len(valid_loader.dataset)
            # accuracy = (TP + TN) / (TP + TN + FP + FN)  # Calculate accuracy based on the formula
            # print(f"Epoch {epoch + 1}, Test Loss: {test_loss}, Accuracy: {accuracy}")
            print("Epoch : {}, Test Loss : {}".format(epoch + 1, test_loss))
            print("Accuracy : {}".format(correct / (len(valid_loader.dataset))))


