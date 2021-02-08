import numpy as np
import matplotlib.pyplot as plt


# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y)
plt.plot([0, 0], [1.0, 0.0], ':')
plt.title('Sigmoid Function')
plt.show()


# Hyperbolic tangent function
y = np.tanh(x)

plt.plot(x, y)
plt.plot([0, 0], [1.0, -1.0], ':')
plt.axhline(y=0, color='orange', linestyle='--')
plt.title('Tanh Function')
plt.show()


# ReLU
def relu(x):
    return np.maximum(0, x)


y = relu(x)

plt.plot(x, y)
plt.plot([0, 0], [5.0, 0.0], ':')
plt.title('ReLU Function')
plt.show()


# Leaky ReLU
def leaky_relu(x, a):
    return np.maximum(a * x, x)


y = leaky_relu(x, 0.1)

plt.plot(x, y)
plt.plot([0, 0], [5.0, 0.0], ':')
plt.title('Leaky ReLU Function')
plt.show()


# Softmax function
y = np.exp(x) / np.sum(np.exp(x))

plt.plot(x, y)
plt.title('Softmax Function')
plt.show()

'''
은닉층: ReLU 또는 그것의 변형판 사용
출력층: 
    이진 분류: Sigmoid => nn.BCELoss()
    다중 클래스 분류: Softmax => nn.CrossEntropyLoss()
    회귀: <NULL> => MSE
'''
