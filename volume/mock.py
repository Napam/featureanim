from torch import nn
from itertools import chain

print(*chain.from_iterable([(nn.Linear(i,i-2, bias=False),nn.ReLU(),nn.BatchNorm1d(i-2)) for i in range(128, 2, -2)]))