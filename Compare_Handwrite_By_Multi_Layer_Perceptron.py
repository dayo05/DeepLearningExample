import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


digits = load_digits()
print('Count of samples: {}'.format(len(digits.images)))

X = digits.data
y = digits.target

import torch
import torch.nn as nn
from torch import optim


model = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 10)
)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.int64)

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters())

losses = []

for epoch in range(100):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, 100, loss.item()
        ))
    losses.append(loss.item())

plt.plot(losses)
plt.show()
