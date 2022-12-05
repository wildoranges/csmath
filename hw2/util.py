import pandas as pd
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from time import strftime

col_names=[i for i in range(42)]

def process_data(train_file='dataset/kddcup99_train.csv', test_file='dataset/kddcup99_test.csv', 
                 train_header=None, test_header=None, names=col_names, 
                 train_label_str='normal.', test_label_str='normal.'):
    train_df = pd.read_csv(train_file, header=train_header, names=names, on_bad_lines='skip')
    test_df = pd.read_csv(test_file, header=test_header, names=names, on_bad_lines='skip')
    
    train_df[41] = (train_df[41] != train_label_str)
    test_df[41] = (test_df[41] != test_label_str)
    label_dict = {True:1, False:0}
    train_df[41].replace(label_dict, inplace=True)
    test_df[41].replace(label_dict, inplace=True)
    
    for column in train_df.columns:
        if column == 1 or column == 2 or column == 3:
            frame = pd.concat((train_df[column], test_df[column]))
            _, map = frame.factorize()
            item_map = {map[i]:i for i in range(len(map))}
                
            train_df[column].replace(item_map, inplace=True)
            test_df[column].replace(item_map, inplace=True)
            
    train_features, train_label = train_df.iloc[:,:-1], train_df.iloc[:,-1]
    test_features, test_label = test_df.iloc[:,:-1], test_df.iloc[:,-1]
    
    return (train_features, train_label), (test_features, test_label)
            
def pca(n, train_features):
    model = PCA(n_components=n)
    model.fit(train_features)
    return model

def bayes_fit(features, label, clf=GaussianNB()):
    clf.fit(features, label)
    return clf

def lower_bayes_fit_test(dim, model, train_features, train_label, 
                  test_features, test_label, prefix, clf=GaussianNB()):
    lower_features = None
    if prefix.upper() == 'AE':
        train_features = torch.tensor(np.array(train_features))
        lower_features = model(train_features).detach().numpy()
    else:
        lower_features = model(train_features)
    
    if dim == 2:
        normal_features = []
        attack_features = []
        for i in range(len(train_label)):
            if train_label[i] == 0:
                normal_features.append(lower_features[i])
            else:
                attack_features.append(lower_features[i])
        normal_features = np.stack(normal_features)
        attack_features = np.stack(attack_features)
        normal_xs = normal_features[:,0]
        normal_ys = normal_features[:,1]
        attack_xs = attack_features[:,0]
        attack_ys = attack_features[:,1]
        plt.scatter(normal_xs, normal_ys, c='blue', label='normal')
        plt.scatter(attack_xs, attack_ys, c='red', label='attack')
        plt.legend()
        plt.savefig('{}_2_dim.jpg'.format(prefix))
        
    bayes_model = bayes_fit(lower_features, train_label, clf)
    lower_test_features = None
    if prefix.upper() == 'AE':
        test_features = torch.tensor(np.array(test_features))
        lower_test_features = model(test_features).detach().numpy()
    else:
        lower_test_features = model(test_features)
    pred = bayes_model.predict(lower_test_features)
    precision, recall, f1score, _ = precision_recall_fscore_support(test_label, pred)
    acc = accuracy_score(test_label, pred)
    print("after lower to dim {}: precision: {} \nrecall: {} \nf1score: {} \naccuracy: {}".format(dim, precision, recall, f1score, acc))

class AutoEncoder(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(41, n, dtype=float),
            torch.nn.Tanh()
        )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(n, 41, dtype=float),
            torch.nn.Tanh()
        )
        
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
def trainAE(model, dataloader, optimizer, epochs, criterion, device):
    model.train()
    for epoch in range(epochs):
        for idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            out_data = model(data)
            loss = criterion(out_data, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if idx % 100 == 0:
                print('Train Epoch: {}/{} [{}/{}]\tLoss: {:.6f}'.format(
                    epoch + 1, epochs, idx * len(data), len(dataloader.dataset), loss.item()))
    
class KddCup99Data(Dataset):
    def __init__(self, train_features, train_label, one_hot=False):
        super().__init__()
        
        self.features = np.array(train_features, dtype=float)
        train_label = np.array(train_label, dtype=float).reshape((train_label.shape[0], 1))
        if one_hot:
            one = np.ones(shape=train_label.shape, dtype=float)
            extra_col = one - train_label
            self.label = np.hstack((train_label, extra_col))
        else:
            self.label =train_label
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index], self.label[index]    

class KddCup99(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.l1 = torch.nn.Sequential(
            torch.nn.Linear(41, 36, dtype=float),
            torch.nn.Tanh()
        )
        
        self.l2 = torch.nn.Sequential(
            torch.nn.Linear(36, 24, dtype=float),
            torch.nn.Tanh()
        )
        
        self.l3 = torch.nn.Sequential(
            torch.nn.Linear(24, 12, dtype=float),
            torch.nn.Tanh()
        )
        
        self.l4 = torch.nn.Sequential(
            torch.nn.Linear(12, 6, dtype=float),
            torch.nn.Tanh()
        )
        
        self.l5 = torch.nn.Sequential(
            torch.nn.Linear(6, 2, dtype=float),
            torch.nn.Tanh()
        )
        
    def forward(self, x):
        return self.l5(self.l4(self.l3(self.l2(self.l1(x)))))
    
def train(model, dataloader, optimizer, epochs, criterion, device):
    model.train()
    for epoch in range(epochs):
        all_loss = []
        for idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            out_target = model(data)
            out_target = out_target.reshape(target.shape)
            loss = criterion(out_target, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if idx % 100 == 0:
                print('Train Epoch: {}/{} [{}/{}]\tLoss: {:.6f}'.format(
                    epoch + 1, epochs, idx * len(data), len(dataloader.dataset), loss.item()))
                all_loss.append(float(loss.item()))
        loss = sum(all_loss) / len(all_loss)
        plt.scatter(epoch + 1, loss)
    plt.xlabel("n_epochs")
    plt.ylabel("loss")
    plt.savefig("./h2_{}.jpg".format(strftime("%m-%d-%H-%M")))
    
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    num_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            out_target = model(data)
            out_target = out_target.reshape(target.shape)
            loss = criterion(out_target, target)
            test_loss += float(loss.item())

            out = torch.softmax(out_target, dim=-1)
            for i in range(out.shape[0]):
                if target[i][0] == 1.0 and out[i][0] >= 0.5:
                    num_correct += 1
                elif target[i][0] == 0.0 and out[i][0] < 0.5:
                    num_correct += 1
                    
        test_loss = test_loss / (len(test_loader))
        accuracy = num_correct / (len(test_loader.dataset))
        print("Test set: Average loss: {:.4f}\tAcc: {:.4f}".format(test_loss, accuracy))
    return (test_loss, )

def train_test(model, dataloader, optimizer, epochs, criterion, test_loader, device):
    epoch_pt = []
    train_loss_pt = []
    test_loss_pt = []
    for epoch in range(epochs):
        model.train()
        all_loss = []
        for idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            out_target = model(data)
            out_target = out_target.reshape(target.shape)
            loss = criterion(out_target, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if idx % 100 == 0:
                print('Train Epoch: {}/{} [{}/{}]\tLoss: {:.6f}'.format(
                    epoch + 1, epochs, idx * len(data), len(dataloader.dataset), loss.item()))
                all_loss.append(float(loss.item()))
        train_loss = sum(all_loss) / len(all_loss)
        test_loss = test(model, test_loader, criterion, device)[0]
        epoch_pt.append(epoch + 1)
        train_loss_pt.append(train_loss)
        test_loss_pt.append(test_loss)
    plt.scatter(epoch_pt, train_loss_pt, c='red', label='train loss')
    plt.scatter(epoch_pt, test_loss_pt, c='blue', label='test loss')
    plt.legend()
    plt.xlabel("n_epochs")
    plt.ylabel("loss")
    time_string = strftime("%m-%d-%H-%M")
    plt.savefig("./h2_{}.jpg".format(time_string))
    print(time_string)
                
                