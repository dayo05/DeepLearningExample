import torch
import torch.optim as optim


torch.manual_seed(1)

# Input
x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])
x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])

# Output
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# H(x) = w1 * x1 + w2 * x2 + w3 * x3 + b
w1 = torch.zeros(1, requires_grad=True)
w2 = torch.zeros(1, requires_grad=True)
w3 = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# Declare optimizer
optimizer = optim.SGD([w1, w2, w3, b], lr=1e-5)

nb_epochs=20000
for epoch in range(nb_epochs):
    # Calculate H(x)
    hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b

    # Calculate cost
    cost = torch.mean((hypothesis - y_train) ** 2)

    # Renew H(x) with cost
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # Create log
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} w1: {:.3f} w2: {:.3f} w3: {:.3f} b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, w1.item(), w2.item(), w3.item(), b.item(), cost.item()
        ))


