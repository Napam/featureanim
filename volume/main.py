from dataload import generator
from net import Mamiew, oneVsAll
from torch.optim import AdamW
import torch
from torch import nn
from tqdm import tqdm
from matplotlib import pyplot as plt 
from sklearn.decomposition import PCA


def train_loop(model, device):
    criterion = oneVsAll
    # criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=5e-3, weight_decay=0.5)

    for epoch in range(100):
        gen = generator(device=device, range_=(0,2048))
        for Xs, ys in tqdm(gen, disable=True):
            pred, latent = model(Xs)
            loss = criterion(pred, ys.ravel())
            print((pred.detach().argmax(1) == ys.ravel()).sum() / len(ys))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    X = PCA(n_components=2).fit_transform(latent.detach().cpu().numpy()).T
    c = ys.detach().cpu().numpy().reshape(-1)
    plt.scatter(*X, c=c)
    plt.savefig("hehe.png")


if __name__ == '__main__':
    device = torch.device('cuda', 1)
    model = Mamiew().to(device)
    train_loop(model, device)