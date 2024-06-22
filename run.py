import time

from train import train
from evaluate import eval
from parser_transformer import args


if __name__ == '__main__':
    start_time = time.time()
    train(args)
    end_time = time.time()
    print("Execution time of train() is: {} seconds".format(end_time - start_time))

    eval(args)
