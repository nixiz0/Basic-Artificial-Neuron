import numpy as np


def log_loss(y_true, y_pred):
    # Small value to avoid logarithms of zero
    epsilon = 1e-15  
    
    # Fix y_pred values ​​to avoid log(0) errors
    y_pred = np.maximum(np.minimum(y_pred, 1 - epsilon), epsilon)
    
    # Calculation of loss for each pair of predicted and actual values
    loss = np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    # Normalization of the average loss
    loss = -loss
    
    return loss

def accuracy_score(y_true, y_pred):
    # Checking list sizes
    if len(y_true) != len(y_pred):
        raise ValueError("The lists of actual and predicted values must have the same length.")
    
    # Calculation of the number of correct predictions
    correct_predictions = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    
    # Calculation of percentage accuracy
    accuracy = correct_predictions / len(y_true) * 100.0
    
    return accuracy