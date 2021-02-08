import torch
import torch.nn as nn


device = 'cpu'
torch.manual_seed(7777)

X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
# XOR
y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)  # Doesn't work
# OR
# y = torch.FloatTensor([[0], [1], [1], [1]]).to(device)

linear = nn.Linear(2, 1, bias=True)
sigmoid = nn.Sigmoid()
model = nn.Sequential(linear, sigmoid).to(device)

criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1)

for step in range(10001):
    optimizer.zero_grad()
    hypothesis = model(X)

    cost = criterion(hypothesis, y)
    cost.backward()
    optimizer.step()

    if step % 100 == 0:
        print(step, cost.item())

with torch.no_grad():
    hypothesis = model(X)
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == y).float().mean()
    print('모델의 출력값(Hypothesis): ', hypothesis.detach().cpu().numpy())
    print('모델의 예측값(Predicted): ', predicted.detach().cpu().numpy())
    print('실제값(Y): ', y.cpu().numpy())
    print('정확도(Accuracy): ', accuracy.item())
