import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(1)

# Input Data
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
# Result Data
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# Declare model
model = nn.Linear(3, 1)

print(list(model.parameters()))

# Define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 50000
for epoch in range(nb_epochs + 1):
    # Calculate H(x)
    prediction = model(x_train)  # Same as model.forward(x_train)

    # Calculate cost
    cost = F.mse_loss(prediction, y_train)  # Default function which provided at pytorch

    # Renew H(x) by cost
    # Reset gradient to 0
    optimizer.zero_grad()
    # Differential cost function to calculate gradient
    cost.backward()
    # Update W, b
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

test_input = torch.FloatTensor([[73, 80, 75]])

pred_y = model(test_input)
print(' result of ', test_input[:, :], ' is ', pred_y[:, 0])
