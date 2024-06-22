import torch.nn as nn
import torch
from models.transformer import TransformerModel
# from parser_my import args
from parser_transformer import args
from dataset import getData
import numpy as np
import random
import matplotlib.pyplot as plt
import time


def train(args):

    myseed = 1337
    # Setup Seeds
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(myseed)
    np.random.seed(myseed)
    random.seed(myseed)

    # model = lstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=2, dropout=args.dropout, batch_first=args.batch_first )
    model = TransformerModel(args.input_size)
    model.to(args.device)
    criterion = nn.CrossEntropyLoss()  # 定义损失函数
    # criterion = nn.BCELoss()  # 定义损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # Adam梯度下降  学习率=0.001

    min_loss = 100
    best_model = None
    best_acc = 0.0

    # Lists to store the loss and accuracy values
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []

    # close_max, close_min, train_loader, test_loader = getData(args.corpusFile,args.sequence_length,args.batch_size )
    close_max, close_min, train_loader, valid_loader, test_loader = getData(args.corpusFile,args.sequence_length,args.batch_size )
    for i in range(args.epochs):
        train_loss = 0.0
        train_acc = 0.0

        model.train()
        for idx, (data, label) in enumerate(train_loader):
            # if args.useGPU:
            #     data1 = data.squeeze(1).cuda()
            #     pred = model(Variable(data1).cuda())
            #     print(pred.shape)
            #     # pred = pred[1,:,:]
            #     label = label.cuda()
            #     # print(label.shape)
            # else:
            #     data1 = data.squeeze(1)
            #     pred = model(Variable(data1))
            #     # pred = pred[1, :, :]
            #     # pred = pred.squeeze(0)
            data = data.squeeze(1).to(args.device)
            label = label.to(args.device)
            pred = model(data)

            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_acc += (pred.argmax(dim=1) == label).sum().item()
            train_loss += loss.item() * data.size(0)
        # Calculate train loss and accuracy
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_acc / len(train_loader.dataset)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        print("Epoch : {}, Train Loss : {}".format(i + 1, train_loss))
        print("Train Accuracy : {}".format(train_acc))

        # if i % 10 == 0:
        #     # torch.save(model, args.save_file)
        #     torch.save({'state_dict': model.state_dict()}, args.save_file)
        #     print('第 %d epoch，保存模型' % i)

        # Validation
        model.eval()  # 声明
        valid_loss = 0.0  # 测试损失
        valid_acc = 0.0

        with torch.no_grad():
            for idx, (x, label) in enumerate(valid_loader):
                x = x.squeeze(1).to(args.device)
                label = label.to(args.device)

                pred = model(x)
                pred = pred.squeeze(0)
                loss = criterion(pred, label)
                valid_loss += loss.item() * x.size(0)
                valid_acc += (pred.argmax(dim=1) == label).sum().item()

        valid_loss = valid_loss / len(valid_loader.dataset)
        valid_acc = valid_acc / len(valid_loader.dataset)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_acc)
        print("Epoch : {}, Valid Loss : {}".format(i + 1, valid_loss))
        print("Valid Accuracy : {}".format(valid_acc))

        # Save the best model
        if valid_loss < min_loss:
            min_loss = valid_loss
            best_model = model
            best_acc = valid_acc
            torch.save({'state_dict': best_model.state_dict()}, args.save_file)
            print("Epoch : {}, saving model with min Loss : {}, best accuracy: {}".format(i + 1, min_loss, best_acc))

    # Plot the loss curve
    fig1 = plt.figure(1)
    plt.plot(range(1, args.epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, args.epochs + 1), valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve (lr={}, batch_size={})'.format(args.lr, args.batch_size))
    plt.savefig('pic/loss_curve_{}_{}.png'.format(args.lr, args.batch_size))  # Save the figure as 'loss_curve.png'
    # plt.show()

    # Plot the accuracy curve
    fig2 = plt.figure(2)
    plt.plot(range(1, args.epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, args.epochs + 1), valid_accuracies, label='Valid Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve (lr={}, batch_size={})'.format(args.lr, args.batch_size))
    plt.savefig('pic/accuracy_curve_{}_{}.png'.format(args.lr, args.batch_size))  # Save the figure as 'accuracy_curve.png'
    # plt.show()


if __name__ == '__main__':
    start_time = time.time()
    train(args)
    end_time = time.time()
    print("Training Execution time of train() is: {} seconds".format(end_time - start_time))
