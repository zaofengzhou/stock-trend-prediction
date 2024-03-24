from LSTMModel import lstm
from dataset import getData
from parser_my import args
import torch
import torch.nn as nn


def eval():
    # model = torch.load(args.save_file)
    model = lstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=2)
    model.to(args.device)
    checkpoint = torch.load(args.save_file)
    model.load_state_dict(checkpoint['state_dict'])
    preds = []
    labels = []

    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()  # 定义损失函数

    close_max, close_min, train_loader, test_loader = getData(args.corpusFile, args.sequence_length, args.batch_size)
    for idx, (x, label) in enumerate(test_loader):
        if args.useGPU:
            x = x.squeeze(1).cuda()  # batch_size,seq_len,input_size
        else:
            x = x.squeeze(1)
        pred = model(x)
        pred = pred.squeeze(0)
        loss = criterion(pred, label)
        test_loss += loss.item() * x.size(0)
        correct += (pred.argmax(dim=1) == label).sum().item()

    test_loss = test_loss / len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)

    print("Test Loss : {}".format(test_loss))
    print("Accuracy : {}".format(acc))

    # for i in range(len(preds)):
    #     print('预测值是%.2f,真实值是%.2f' % (
    #     preds[i][0] * (close_max - close_min) + close_min, labels[i] * (close_max - close_min) + close_min))

        # preds[i] = preds[i][0] * (close_max - close_min) + close_min
        # labels[i] = labels[i] * (close_max - close_min) + close_min

    # import matplotlib.pyplot as plt
    # import numpy as np
    # plt.plot(np.arange(len(preds)), preds, 'r', label='prediction')
    # plt.plot(np.arange(len(labels)), labels, 'b', label='real')
    # plt.show()

eval()