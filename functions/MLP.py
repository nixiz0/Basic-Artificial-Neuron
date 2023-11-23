import numpy as np 
from .constant import *


def initialisation(dim):
    params = {}
    C = len(dim)
    
    for c in range(1, C):
        params['W' + str(c)] = np.random.randn(dim[c], dim[c-1])
        params['b' + str(c)] = np.random.randn(dim[c], 1)

    return params

def forward_propagation(X, params):
    activations = {'A0': X}
    C = len(params) // 2
    
    for c in range(1, C + 1):
        Z = params['W' + str(c)].dot(activations['A' + str(c-1)]) + params['b' + str(c)]
        activations['A' + str(c)] = 1 / (1 + np.exp(-Z))
    
    return activations

def back_propagation(y, activations, params):
    m = y.shape[1]
    C = len(params) // 2
    
    dZ = activations['A' + str(C)] - y 
    gradients = {}
    
    for c in reversed(range(1, C + 1)):
        gradients['dW' + str(c)] = 1 / m * np.dot(dZ, activations['A' + str(c - 1)].T)
        gradients['db' + str(c)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        if c > 1:
            dZ = np.dot(params['W' + str(c)].T, dZ) * activations['A' + str(c-1)] * (1-activations['A' + str(c-1)])
    
    return gradients

def update(gradients, params, lr):
    C = len(params) // 2
    
    for c in range(1, C + 1):
        params['W' + str(c)] -= lr * gradients['dW' + str(c)]
        params['b' + str(c)] -= lr * gradients['db' + str(c)]
    
    return params

def predict(X, params):
    activations = forward_propagation(X, params)
    C = len(params) // 2
    Af = activations['A' + str(C)]
    return (Af >= 0.5).astype(int)

def neural_network(X, y, hidden_layers=(8,8,8), lr=0.1, n_iter=1000):
    dim = list(hidden_layers)
    dim.insert(0, X.shape[0])
    dim.append(y.shape[0])
    params = initialisation(dim)
    
    train_loss = []
    train_acc = []
    
    for i in range(n_iter):
        activations = forward_propagation(X, params)
        gradients = back_propagation(y, activations, params)
        params = update(gradients, params, lr)
        
        if i%100 == 0 :
            C = len(params) // 2
            train_loss.append(log_loss(y, activations['A' + str(C)]))
            y_pred = predict(X, params)
            current_accuracy = accuracy_score(y.flatten(), y_pred.flatten())
            train_acc.append(current_accuracy)
        
        print(f"Train Loss : {train_loss} \nTrain Accuracy : {train_acc}")
    return params


#<--Display Function for Evaluation-->
def display_mlp(X, y, hidden_layers, lr, n_iter):
    # Training the neural network
    trained_params = neural_network(X, y, hidden_layers, lr, n_iter)
    
    # Prediction for XOR data
    predictions = predict(X, trained_params)

    # Accuracy check
    accuracy = accuracy_score(y.flatten(), predictions.flatten())
    print(f"Accuracy: {accuracy}%")

    # Calculation of the number of correct predictions
    num_correct_predictions = np.sum(predictions == y)
    total_predictions = y.size
    accuracy = num_correct_predictions / total_predictions * 100.0
    print(f"\n----Number of correct predictions: {num_correct_predictions} out of {total_predictions}----")