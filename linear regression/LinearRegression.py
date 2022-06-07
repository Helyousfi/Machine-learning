import numpy as np
#Creating the data
X = np.array([2, 4, 5, 6, 8, 9, 3, 7])
Y_real = np.array([4, 8, 10, 12, 16, 18, 6, 14])
w = 0.0
#Defining the forward pass
def forward(X):
    return w*X
#Loss : MSE 
def loss_function(Y_real, Y_predicted):
    return (Y_real - Y_predicted).mean()**2
# MSE = 1/N * (w*X - Y_predicted)**2
# dMSE/dw = 1/N * 2X * (w*X - Y_real)
# we compute the gradient descent to update the value of the weight : W
def gradientDescent(w, X, Y_real):
    dMSE = (2 * X * (w*X - Y_real)).mean()
    return dMSE
nIters = 10 #Number of iterations = epochs
lr = 0.01 #The learning rate
#Print the result before training
print(forward(X))
for epoch in range(nIters):
    # Predict the value of Y
    Y_predicted = forward(X)

    # Compute the loss
    MSE = loss_function(Y_predicted, Y_real) # MSE = mean squared error
    
    # Compute the gradient
    dw = gradientDescent(w, X, Y_real)

    # Updating 
    w = w - lr * dw
    
    if epoch % 1 == 0:
        print(f"Epoch : {epoch}  Value of w is {w}")

#Print the result after training
print(forward(X))


