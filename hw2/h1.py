from util import *
import torch
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

if __name__=='__main__':
    dims = [32, 16, 8, 4, 2]
    (train_features, train_label), (test_features, test_label) = process_data()
    dataset = KddCup99Data(train_features, train_label)
    dataloader = DataLoader(dataset, batch_size=16000, shuffle=True)

    for dim in dims:
        # PCA
        pca_model = pca(dim, train_features)
        lower_bayes_fit_test(dim, pca_model.transform, train_features, train_label, test_features, test_label, 'PCA')
        # AutoEncoder
        model = AutoEncoder(dim).to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(),
                                lr = 1e-5)
        trainAE(model, dataloader, optimizer, 5, criterion, device)
        torch.save(model, 'h1_AE_Dim{}.pt'.format(dim))
        model = model.to('cpu')
        model.eval()
        lower_bayes_fit_test(dim, model.encoder, train_features, train_label, test_features, test_label, 'AE')
        
        