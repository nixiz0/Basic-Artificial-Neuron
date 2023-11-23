import numpy as np
from .constant import *


def initialisation(X):
    W = np.random.randn(X.shape[0], 1)
    b = np.random.randn(1)
    return W, b

def model(X, W, b):
    Z = np.dot(W.T, X) + b
    A = 1 / (1 + np.exp(-Z))
    return A

def gradients(A, X, y):
    dZ = A - y
    dW = np.dot(X, dZ.T) / X.shape[1]
    db = np.sum(dZ) / X.shape[1]
    return dW, db

def update(dW, db, W, b, lr):
    W -= lr * dW
    b -= lr * db
    return W, b

def predict(X, W, b):
    A = model(X, W, b)
    return A >= 0.5

def artificial_neuron(X, y, lr, n_iter):
    W, b = initialisation(X)

    loss = []

    for i in range(n_iter):
        A = model(X, W, b)
        loss.append(log_loss(y, A))
        dW, db = gradients(A, X, y)
        W, b = update(dW, db, W, b, lr)

        if i % 10 == 0:
            print(f"Loss of the model : {loss[-1]}")
            
    return W, b


#<--Display Function for Evaluation-->
def display_artificial_neuron(X, y, lr, n_iter):    
    # Training the neural network
    trained_params = artificial_neuron(X, y, lr, n_iter)

    # Prediction for XOR data
    predictions = predict(X, *trained_params)

    # Accuracy check
    accuracy = accuracy_score(y.flatten(), predictions.flatten())
    print(f"Accuracy: {accuracy}%")

    # Calculation of the number of correct predictions
    num_correct_predictions = np.sum(predictions == y)
    total_predictions = y.size
    print(f"\n----Number of correct predictions: {num_correct_predictions} out of {total_predictions}----")