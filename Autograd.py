import torch


w = torch.tensor(2.0, requires_grad=True)

y = w ** 2
z = 2 * y + 5  # z = w ** 2 + z
z.backward()  # z' = 4 * w
print('Result:', format(w.grad))
