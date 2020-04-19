import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F

###
path_prefix = './'

def load_training_data(path='training_label.txt'):
    # 把 training 時需要的 data 讀進來
    # 如果是 'training_label.txt'，需要讀取 label，如果是 'training_nolabel.txt'，不需要讀取 label
    if 'training_label' in path:
        with open(path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip('\n').split(' ') for line in lines]
        x = [line[2:] for line in lines]
        for line in x:
            for word in line:
                if word.islower() == False:
                    line.remove(word)
        y = [line[0] for line in lines]
        return x, y
    else:
        with open(path, 'r') as f:
            lines = f.readlines()
            x = [line.strip('\n').split(' ') for line in lines]
            for line in x:
                for word in line:
                    if word.islower() == False:
                        line.remove(word)
        return x

def load_testing_data(path='testing_data'):
    # 把 testing 時需要的 data 讀進來
    with open(path, 'r') as f:
        lines = f.readlines()
        X = ["".join(line.strip('\n').split(",")[1:]).strip() for line in lines[1:]]
        X = [sen.split(' ') for sen in X]
        for line in X:
            for word in line:
                if word.islower() == False:
                    line.remove(word)
    return X        


def evaluation(outputs, labels):
    # outputs => probability (float)
    # labels => labels
    outputs[outputs>=0.5] = 1 # 大於等於 0.5 為有惡意
    outputs[outputs<0.5] = 0 # 小於 0.5 為無惡意
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct