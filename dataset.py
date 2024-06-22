from pandas import read_csv
import numpy as np
from torch.utils.data import DataLoader,Dataset
import torch
from torchvision import transforms
# from parser_my import args
from parser_transformer import args
import yfinance as yf
import tushare as ts
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
from collections import Counter


def getDataFromTushare(start_date, end_date):
    ts.set_token('dd635f1da821210b53e89815c68b5469fef4dfc2cc33bbed14e1c6ec')
    pro = ts.pro_api()
    today = datetime.datetime.today().strftime('%Y%m%d')    #获取今天的年月日
    lastday = datetime.datetime.today() - datetime.timedelta(days=1)  #获取前一天数据
    lastday = lastday.strftime('%Y%m%d')
    last_year = datetime.datetime.today() - relativedelta(months=60)   #获取前一年的日期
    last_year = last_year.strftime('%Y%m%d')   # 转换成STR
    Lastweek = datetime.datetime.today() - datetime.timedelta(days=7)   #获取前一周的日期
    Lastweek = Lastweek.strftime('%Y%m%d')
    # df = pro.daily(ts_code='600519.SH', start_date=last_year, end_date=today)
    df = pro.daily(ts_code='600519.SH', start_date=start_date, end_date=end_date)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df.set_index('trade_date', inplace=True)
    df = df.sort_values(by='trade_date')
    # plt.figure(figsize=(16, 8))
    # plt.figure(figsize=(14, 7))
    # plt.plot(df['close'])
    # plt.show()
    df = df[['open', 'high', 'low', 'close', 'vol']]
    return df


def getDataFromYFinance(stock_ticker):
    nasdaq_ticker = stock_ticker
    stock_data = yf.download(nasdaq_ticker, start='2001-11-22', end='2021-11-20')
    stock_data.dropna(inplace=True)  # 删除缺失值
    stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    return stock_data


def getDataFromFile(corpusFile):
    stock_data = read_csv(corpusFile)
    # stock_data.drop('ts_code', axis=1, inplace=True)  # 删除第二列’股票代码‘
    # stock_data.drop('id', axis=1, inplace=True)  # 删除第一列’id‘
    # stock_data.drop('pre_close', axis=1, inplace=True)  # 删除列’pre_close‘
    # stock_data.drop('trade_date', axis=1, inplace=True)  # 删除列’trade_date‘
    # df = stock_data.copy().drop(['Tomorrow Trend', 'trade_date'], axis=1)
    df = stock_data.copy().drop(['trade_date'], axis=1)
    return df


def getData(corpusFile,sequence_length,batchSize):
    # 数据预处理 ，去除id、股票代码、前一天的收盘价、交易日期等对训练无用的无效数据
    # stock_data = getDataFromFile(corpusFile)

    nasdaq_ticker = "^IXIC"
    # stock_data = getDataFromYFinance(nasdaq_ticker)

    start_date = '20190520'
    end_date = '20240430'
    stock_data = getDataFromTushare(start_date, end_date)

    stock_data['labels'] = np.where(stock_data['close'].shift(-1) >= stock_data['close'], 1, 0)

    close_max = stock_data['close'].max()  # 收盘价的最大值
    close_min = stock_data['close'].min()  # 收盘价的最小值
    df = stock_data.copy().drop(['labels'], axis=1)
    df = df.apply(lambda x: (x - min(x)) / (max(x) - min(x)))  # min-max标准化

    # 构造X和Y
    #根据前n天的数据，预测未来一天的收盘价(close)， 例如：根据1月1日、1月2日、1月3日、1月4日、1月5日的数据（每一天的数据包含8个特征），预测1月6日的收盘价。
    sequence = sequence_length
    X = []
    Y = []
    for i in range(df.shape[0] - sequence):
        X.append(np.array(df.iloc[i:(i + sequence), ].values, dtype=np.float32))
        # Y.append(labels[i + sequence - 1])
        Y.append(stock_data['labels'].iloc[i + sequence - 1])

    # 构建batch
    total_len = len(Y)
    # print(total_len)

    train_len = int(0.8 * total_len)
    valid_len = int(0.1 * total_len)
    test_len = total_len - train_len - valid_len
    print("train_len : {}, valid_len: {}, test_len: {}".format(train_len, valid_len, test_len))

    trainx, trainy = X[:train_len], Y[:train_len]
    validx, validy = X[train_len:train_len + valid_len], Y[train_len:train_len + valid_len]
    testx, testy = X[-test_len:], Y[-test_len:]

    counts_train = Counter(trainy)
    total_train = len(trainy)
    percentage_train = {k: v / total_train for k, v in counts_train.items()}
    print(percentage_train)

    counts_valid = Counter(validy)
    total_valid = len(validy)
    percentage_valid = {k: v / total_valid for k, v in counts_valid.items()}
    print(percentage_valid)

    counts_test = Counter(testy)
    total_test = len(testy)
    percentage_test = {k: v / total_test for k, v in counts_test.items()}
    print(percentage_test)

    # trainx, testx, trainy, testy = train_test_split(X, Y, test_size=0.3, random_state=42)
    # trainx, validx, trainy, validy = train_test_split(trainx, trainy, test_size=0.1, random_state=42)

    train_loader = DataLoader(dataset=Mydataset(trainx, trainy, transform=transforms.ToTensor()), batch_size=batchSize,
                              shuffle=True)
    valid_loader = DataLoader(dataset=Mydataset(validx, validy, transform=transforms.ToTensor()), batch_size=batchSize,
                              shuffle=False)
    test_loader = DataLoader(dataset=Mydataset(testx, testy), batch_size=batchSize, shuffle=False)
    return close_max,close_min,train_loader, valid_loader, test_loader


class Mydataset(Dataset):
    def __init__(self, xx, yy, transform=None):
        self.x = xx
        self.y = yy
        self.tranform = transform

    def __getitem__(self, index):
        x1 = self.x[index]
        y1 = self.y[index]
        # y1 = np.int64(y1)
        y1 = torch.tensor(y1, dtype=torch.long)
        if self.tranform != None:
            x1 = self.tranform(x1)
        return x1, y1

    def __len__(self):
        return len(self.x)


if __name__ == '__main__':
    close_max, close_min, train_loader, valid_loader, test_loader = getData(args.corpusFile, args.sequence_length, args.batch_size)
    # debug train_loader
    for i, (x, y) in enumerate(train_loader):
        print(i)
        print(x.shape)
        print(y)
        break