import tkinter as tk
from tkinter import ttk, simpledialog
import numpy as np
from tkinter_fct import *
from functions.Artificial_Neuron import display_artificial_neuron
from functions.MLP import display_mlp


def open_artificial_neuron():
    app_theme(root)    
    #<---Dataset XOR--->
    X = np.array([[0, 0],[0, 1],[1, 0],[1, 1],[0.5, 0.5],
                    [0.3, 0.7],[0.8, 0.1],[0.9, 0.9]])

    y = np.array([[0],[1],[1],[0],[0],[1],[1],[0]])
    X = X.T
    y = y.T

    #<---User Input--->
    lr = simpledialog.askfloat("Learning Rate", "Enter the learning rate :")
    n_iter = simpledialog.askinteger("Iterations", "Enter the number of iterations :")
        
    #<---Neuron Function--->
    if lr is not None and n_iter is not None and lr != 0 and n_iter != 0:
        display_artificial_neuron(X, y, lr, n_iter)
        

def open_MLP():
    app_theme(root)
    #<---Dataset XOR--->
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.2, 0.3], 
                  [0.8, 0.9], [0.1, 0.9], [0.7, 0.2], [0.4, 0.6], [0.5, 0.5]]).T
    
    y = np.array([[0, 1, 1, 0, 1, 0, 1, 0, 1, 0]])

    #<---User Input--->
    hidden_layers = simpledialog.askstring("Hidden Layers", "Number of neurons and layers you want (ex: 8,8,8):")
    lr = simpledialog.askfloat("Learning Rate", "Enter the learning rate :")
    n_iter = simpledialog.askinteger("Iterations", "Enter the number of iterations :")
        
    #<---Neuron Function--->
    if hidden_layers and lr and n_iter is not None and hidden_layers and lr and n_iter != 0:
        hidden_layers = tuple(map(int, hidden_layers.split(',')))
        display_mlp(X, y, hidden_layers, lr, n_iter)
        

# Creation of the main window
root = tk.Tk()
root.title("Main Interface")

# Minimum and maximum window size
root.minsize(200, 150)
root.maxsize(300, 180)

# Configuring the size of the main window
root.geometry("250x160")

# Change background color of the main window
root.configure(bg='#333333')

# Centering the main window
center_window(root)

# Creating a style for the button
style = ttk.Style()
style.configure('Custom.TButton', font=('Inter', 16, 'bold'), foreground='#001d3d', background='#023e8a')

# Creation of the button for the "Artificial Neuron" interface with custom style
btn_artificial_neuron = ttk.Button(root, text="Artificial Neuron", command=open_artificial_neuron, style='Custom.TButton')
btn_artificial_neuron.pack(pady=20)

# Creation of the button for the "MLP" interface with the custom style
btn_MLP = ttk.Button(root, text="MLP", command=open_MLP, style='Custom.TButton')
btn_MLP.pack(pady=5)

# Launching the main interface loop
root.mainloop()