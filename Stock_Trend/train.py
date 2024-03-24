from torch.autograd import Variable
import torch.nn as nn
import torch
from LSTMModel import lstm
from parser_my import args
from dataset import getData

def train():

    model = lstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=2, dropout=args.dropout, batch_first=args.batch_first )
    model.to(args.device)
    criterion = nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # Adam梯度下降  学习率=0.001

    min_loss = 100
    best_model = None
    best_acc = 0

    close_max, close_min, train_loader, test_loader = getData(args.corpusFile,args.sequence_length,args.batch_size )
    for i in range(args.epochs):
        total_loss = 0
        for idx, (data, label) in enumerate(train_loader):
            if args.useGPU:
                data1 = data.squeeze(1).cuda()
                pred = model(Variable(data1).cuda())
                # print(pred.shape)
                pred = pred[1,:,:]
                label = label.unsqueeze(1).cuda()
                # print(label.shape)
            else:
                data1 = data.squeeze(1)
                pred = model(Variable(data1))
                # pred = pred[1, :, :]
                pred = pred.squeeze(0)
                # label = label
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
        total_loss = total_loss / len(train_loader.dataset)
        print("Epoch : {}, Train Loss : {}".format(i + 1, total_loss))

        if i % 10 == 0:
            # torch.save(model, args.save_file)
            torch.save({'state_dict': model.state_dict()}, args.save_file)
            print('第 %d epoch，保存模型' % i)
        # torch.save(model, args.save_file)

        model.eval()  # 声明
        correct = 0.0  # 计算正确率
        test_loss = 0.0  # 测试损失

        with torch.no_grad():
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

        if test_loss < min_loss:
            min_loss = test_loss
            best_acc = acc
            best_model = {'state_dict': model.state_dict()}
            print("Epoch : {}, min Loss : {}, best accuracy: {}".format(i + 1, min_loss, best_acc))

        print("Epoch : {}, Test Loss : {}".format(i + 1, test_loss))
        print("Accuracy : {}".format(acc))

    print("Min Loss : {}, best accuracy : {}".format(min_loss, best_acc))
    torch.save(best_model, args.save_file)

train()