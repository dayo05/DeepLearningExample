import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# Data
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[3], [5], [7]])

model = nn.Linear(1, 1)

print(list(model.parameters()))

# Declare optimizer which uses SGD
# learning rate is 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for test in range(100):
    # Repeat this for 2000 times
    nb_epochs = 200
    for epoch in range(nb_epochs + 1):
        # Calculate H(x)
        prediction = model(x_train)

        # Calculate cost
        cost = F.mse_loss(prediction, y_train)  # This is default loss function in pytorch

        # Renew cost using H(x)
        # Reset gradient to 0
        optimizer.zero_grad()
        # Calculate gradient by differential cost function
        cost.backward()  # backward function
        # Update W, b
        optimizer.step()

        if epoch % 100 == 0:
            # Print log
            print('Epoch {:4d}/{} Cost: {:.6f}'.format(
                epoch, nb_epochs, cost.item()
            ))

    # Manual input
    new_var = torch.FloatTensor([[4.0], [10.0]])
    # Get predict y and save to pred_y
    pred_y = model(new_var)  # Forward
    print('In TC ', test, ' result of ', new_var[:, 0], ' is ', pred_y[:, 0])
