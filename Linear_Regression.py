import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_train = torch.FloatTensor([[1], [2], [3]])  # Input
y_train = torch.FloatTensor([[2], [4], [6]])  # Output

print(x_train)
print(x_train.shape)

print(y_train)
print(y_train.shape)

# H(x) = W * x + b
W = torch.zeros(1, requires_grad=True)
print(W)
b = torch.zeros(1, requires_grad=True)

nb_epochs = 20000
for epoch in range(nb_epochs + 1):
    hypothesis = x_train * W + b

    # Get cost function
    cost = torch.mean((hypothesis - y_train) ** 2)

    # Declare optimizer
    optimizer = optim.SGD([W, b], lr=0.01)
    optimizer.zero_grad()

    # Calculate gradient
    cost.backward()

    # Update value
    optimizer.step()

    # Create log
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))
