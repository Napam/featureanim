from dataload import generator
from net import Benja, oneVsAll
from torch.optim import AdamW
import torch
from torch import nn
from tqdm import tqdm
from matplotlib import pyplot as plt 


def train_loop(model, device):
    criterion = oneVsAll
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    for epoch in range(50):
        gen = generator(device=device)
        for Xs, ys in tqdm(gen, disable=True):
            pred, latent = model(Xs)
            loss = criterion(pred, ys.ravel())
            print((pred.detach().argmax(1) == ys.ravel()).sum() / len(ys))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    X = latent.detach().cpu().numpy().T
    c = ys.detach().cpu().numpy().reshape(-1)
    plt.scatter(*X, c=c)
    plt.savefig("hehe.png")


if __name__ == '__main__':
    device = torch.device('cuda', 1)
    model = Benja().to(device)
    train_loop(model, device)