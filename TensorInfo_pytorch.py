import torch

t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print(t)

print('Dim of t =', t.dim())
print('Size of t =', t.size())

print('t[0], t[1], t[-1] =', t[0], t[1], t[-1])
print('t[2:5] t[4:-1] =', t[2:5], t[4:-1])
print('t[:2] t[3:] =', t[:2], t[3:])
print('t[:] =', t[:])

t = torch.FloatTensor([[1., 2., 3.],
                       [4., 5., 6.],
                       [7., 8., 9.],
                       [10., 11., 12.]
                       ])
print(t)

print('Dim of t =', t.dim())
print('Size of t =', t.size())

print('t[:, 1] =', t[:, 1])
print('t[:, 1].size() =', t[:, 1].size())
print('t[:, :-1] =', t[:, :-1])

m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])
print(m1, '+', m2, '=', m1 + m2)

# This is incorrect in math but able to do in pytorch
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([3])  # Convert to [3, 3]
print(m1, '+', m2, '=', m1 + m2)


# Matrix multiplication
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape)
print('Shape of Matrix 2: ', m2.shape)
print(m1.matmul(m2))

# Element multiplication
print(m1 * m2)  # m2 will convert to [[1, 1], [2, 2]] automatically


# Mean
t = torch.FloatTensor([1, 4, 6])
print(t.mean())

t = torch.FloatTensor([[1, 2], [3, 4]])
print(t.mean())

print(t.mean(dim=0))  # Result: {mean of [1, 3]} {mean of [2, 4]}
print(t.mean(dim=1))  # Result: {mean of [1, 2]} {mean of [3, 4]}
print(t.mean(dim=-1))  # Result: same with last dim removed result


# Max
print(t.max())
print(t.max(dim=0))  # Result: not only max and also argmax
print(t.max(dim=0)[0])  # Get only max
print(t.max(dim=0)[1])  # Get only argmax


# Resize(View)
ft = torch.FloatTensor([[[0, 1, 2], [3, 4, 5]],
                        [[6, 7, 8], [9, 10, 11]]])
print(ft.size())
print(ft.view([-1, 3]))  # Convert size to (auto, 3)
print(ft.view([-1, 3]).shape)
print(ft.view([-1, 1, 3]))
print(ft.view([-1, 1, 3]).shape)


# Squeeze(Remove dim which size is 1)
ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.size())
print(ft.squeeze())
print(ft.squeeze().shape)


# UnSqueeze(Add size 1 dim to somewhere)
ft = torch.Tensor([0, 1, 2])
print(ft.size())
print(ft.unsqueeze(0))  # Convert to (1, 3) vector, Result: [[0, 1, 2]]
print(ft.unsqueeze(0).shape)
# Same with
print(ft.view(1, -1))
print(ft.view(1, -1).shape)


# Type casting
bt = torch.ByteTensor([True, False, False, True])
print(bt)
print(bt.long())
print(bt.float())


# Cat tensor
x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])
print(torch.cat([x, y], dim=0))  # Result: [[1, 2], [3, 4], [5, 6], [7, 8]]
print(torch.cat([x, y], dim=1))  # Result: [[1, 2, 5, 6], [3, 4, 7, 8]]

# Stacking
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])
print(torch.stack([x, y, z]))  # Result: [[1, 4], [2, 5], [3, 6]], Same with torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)])
print(torch.stack([x, y, z], dim=1))  # Result: [[1, 2, 3], [4, 5, 6]]


# ones_like, zeros_like
x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(torch.ones_like(x))
print(torch.zeros_like(x))


# Override calculation
x = torch.FloatTensor([[1, 2], [3, 4]])
print(x.mul(2))
print(x)
print(x.mul_(2))
print(x)
