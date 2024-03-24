import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import quandl
import numpy as np
from datetime import date

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 假设 x_train 和 y_train 是 PyTorch 的张量
# x_train 的 shape 为 (样本数, 时间步数, 特征数)
# y_train 的 shape 为 (样本数,)


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


if __name__ == '__main__':
    ss_index = pd.read_csv("/Users/zfzhou/Downloads/000001.SS.csv", engine="python")
    print(ss_index.shape)
    # ss_index.tail()
    ss_index.head()

    missing_values_rows = ss_index[ss_index.isnull().any(axis=1) | (ss_index == 0).any(axis=1)]

    ss_index = ss_index.drop(missing_values_rows.index)

    ss_index['Date'] = pd.to_datetime(ss_index['Date'])
    ss_index.set_index('Date', inplace=True)

    plt.figure(figsize=(16, 8))
    # plt.figure(figsize=(14, 7))
    plt.plot(ss_index['Close'])
    plt.show()

    # 时间点长度
    time_stamp = 50

    train_len = int(len(ss_index) * 0.7)

    # 划分训练集与验证集
    ss_index = ss_index[['Open', 'High', 'Low', 'Close', 'Volume']]  # 'Volume'
    train = ss_index[0:train_len + time_stamp]
    valid = ss_index[train_len - time_stamp:]

    train_labels = [-1]
    previous_close = None
    for index, row in train.iterrows():
        if previous_close is not None:
            if row['Close'] > previous_close:
                train_labels.append(1)
            else:
                train_labels.append(0)
        previous_close = row['Close']

    valid_labels = [-1]
    previous_close = None
    for index, row in valid.iterrows():
        if previous_close is not None:
            if row['Close'] > previous_close:
                valid_labels.append(1)
            else:
                valid_labels.append(0)
        previous_close = row['Close']

    pd.value_counts(train_labels)
    pd.value_counts(valid_labels)

    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(train)
    x_train, y_train = [], []

    # 训练集
    for i in range(time_stamp, len(train)):
        x_train.append(scaled_data[i - time_stamp:i])
        y_train.append(train_labels[i])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # 验证集
    scaled_data = scaler.fit_transform(valid)
    x_valid, y_valid = [], []
    for i in range(time_stamp, len(valid)):
        x_valid.append(scaled_data[i - time_stamp:i])
        y_valid.append(valid_labels[i])

    x_valid, y_valid = np.array(x_valid), np.array(y_valid)

    # 假设 x_train 和 y_train 是 PyTorch 的张量
    # x_train 的 shape 为 (样本数, 时间步数, 特征数)
    # y_train 的 shape 为 (样本数,)
    epochs = 3
    batch_size = 16

    # 转换 x_train 和 y_train 为 TensorDataset
    x_train_tensor = torch.FloatTensor(x_train)
    y_train_tensor = torch.LongTensor(y_train)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    x_valid_tensor = torch.FloatTensor(x_valid)
    y_valid_tensor = torch.LongTensor(y_valid)

    valid_dataset = TensorDataset(x_valid_tensor, y_valid_tensor)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    input_size = x_train.shape[-1]
    hidden_size = 100
    num_layers = 1
    model = LSTM(input_size, hidden_size, num_layers)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters())

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
        print(f"Epoch {epoch + 1}, Loss: {train_loss}")

        model.eval()  # 声明
        correct = 0.0  # 计算正确率
        test_loss = 0.0  # 测试损失
        with torch.no_grad():
            for inputs, labels in valid_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
        test_loss = test_loss / len(valid_loader.dataset)
        accuracy = correct / len(valid_loader.dataset)
        print(f"Epoch {epoch + 1}, Test Loss: {test_loss}, Accuracy: {accuracy}")


    # model.eval()
    # closing_price = model(torch.from_numpy(x_valid).to(torch.float32))
    # scaler.fit_transform(pd.DataFrame(valid['Close'].values))
    # # 反归一化
    # closing_price = scaler.inverse_transform(closing_price.detach().numpy())
    # y_valid = scaler.inverse_transform([y_valid])
    #
    # rms = np.sqrt(np.mean(np.power((y_valid - closing_price), 2)))
    # print(rms)
    # print(closing_price.shape)
    # print(y_valid.shape)
    #
    # plt.figure(figsize=(16, 8))
    # dict_data = {
    #     'Predictions': closing_price.reshape(1, -1)[0],
    #     'Close': y_valid[0]
    # }
    # data_pd = pd.DataFrame(dict_data)
    #
    # plt.plot(data_pd[['Close', 'Predictions']])
    # plt.show()
