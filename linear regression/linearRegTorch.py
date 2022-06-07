"""
y = w1*x1 + w2*x2 + b
w1 = 2, w2 = 3, b = 1
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

X1 = np.array([1, 2, 3, 4, 1.1, 1.3, 2.4, 7], dtype = np.float32)
X2 = np.array([4, 3, 2, 1.5, 1, 1, 2, 2], dtype = np.float32)
Y_real = np.array([15, 14, 13, 13.5, 6.2, 6.6, 13.8, 21], dtype = np.float32)

X1 = torch.from_numpy(X1)
X2 = torch.from_numpy(X2)
Y_real = torch.from_numpy(Y_real)

w1 = torch.tensor(0.0, dtype = torch.float32, requires_grad=True)
w2 = torch.tensor(0.0, dtype = torch.float32, requires_grad=True)
b1 = torch.tensor(0.0, dtype = torch.float32, requires_grad=True)

#forward pass
def forward(X1, X2):
    Y_predicted = w1 * X1 + w2 * X2 + b1
    return Y_predicted

def loss(Y_real, Y_predicted):
    return ((Y_real - Y_predicted)**2).mean()

epochs = 50
lr = 0.01
loss_array = []

for epoch in range(epochs):
    Y_predicted = forward(X1, X2)
    MSE = loss(Y_real, Y_predicted)
    MSE.backward()
    loss_array.append(MSE)
    with torch.no_grad():
        w1.sub_(lr*w1.grad)
        w2.sub_(lr*w2.grad)
        b1.sub_(lr*b1.grad) 
    w1.grad.zero_()
    w2.grad.zero_()
    b1.grad.zero_()
    if(epoch % 1 == 0):
        print(f"Epoch : {epoch} Loss : {MSE:.3f}")

plt.plot(loss_array)
plt.title("Loss over iterations")
plt.show()

#print(w1, w2, b1)
#print(f"Forward = {forward(X1,X2).detach().numpy()}")

def randrange(n, vmin, vmax):
    """
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    """
    return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

n = 100
# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].

xs = randrange(n, 0, 10)
ys = randrange(n, 0, 10)
zs = forward(torch.from_numpy(xs), torch.from_numpy(ys))
zs = zs.detach()
zs = zs.numpy()
print(f"zs = {zs}")
ax.scatter(xs, ys, zs)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

