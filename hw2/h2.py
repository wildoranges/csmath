from util import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import random
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

if __name__=='__main__':
    lines = []
    with open('dataset/kddcup99/data/kddcup99_csv.csv') as f:
        lines = f.readlines()
        
    header_line = lines[0]
    lines = lines[1:]
    random.shuffle(lines)
    length = len(lines)
    train_lines = lines[:length//2]
    test_lines = lines[length//2:]
    
    with open('train.csv', 'w+') as f:
        f.writelines(train_lines)
    with open('test.csv', 'w+') as f:
        f.writelines(test_lines)
    
    (train_features, train_label), (test_features, test_label) = process_data(train_file='train.csv', test_file='test.csv', train_label_str='normal', test_label_str='normal')
    dataset = KddCup99Data(train_features, train_label, one_hot=True)
    dataloader = DataLoader(dataset, batch_size=10000, shuffle=True)
    model = KddCup99().to(device)
    # init = nn.init.constant_
    init = nn.init.normal_
    # init = nn.init.xavier_uniform_
    # init = nn.init.kaiming_normal_
    for m in model.modules():
        if isinstance(m, (nn.Linear, )):
            init(m.weight)
            
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(),
    #                             lr = 1e-2)
    optimizer = torch.optim.Adam(model.parameters(),
                                lr = 1e-3)
    # train(model, dataloader, optimizer, 10, criterion)
    # torch.save(model, 'h2_model.pt')
    # model = torch.load('h2_model.pt')
    # criterion = torch.nn.BCELoss()
    test_dataset = KddCup99Data(test_features, test_label, one_hot=True)
    test_loader = DataLoader(test_dataset, batch_size=10000)
    train_test(model, dataloader, optimizer, 10, criterion, test_loader, device)