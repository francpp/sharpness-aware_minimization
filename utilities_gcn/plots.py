import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import json

def load_data(filename):
    """ Open a json file and save it as a list of dictionaries"""
    # Open the file for reading
    with open(filename, 'r') as f:

        # Initialize an empty list to store the dictionaries
        data = []

        # Iterate over the lines in the file
        for line in f:

            # Parse the line as a dictionary
            d = json.loads(line)

            # Append the dictionary to the list
            data.append(d)
    
    return data
        

def record_stats(data):
    
    losses_train = []
    losses_test = []
    accuracies_train = []
    accuracies_test = []
    epochs = []
    
    for i in range(len(data)):
        if data[i]['step_type'] == 'train':
            losses_train.append(data[i]['loss'])
            accuracies_train.append(data[i]['accuracy'])
            epochs.append(data[i]['epoch'])

        if data[i]['step_type'] == 'test':
            losses_test.append(data[i]['loss'])
            accuracies_test.append(data[i]['accuracy'])
            
    return losses_train, losses_test, accuracies_train, accuracies_test, epochs
            
def plot_loss(epochs, losses_test):

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(epochs, losses_test)
    ax.set_title('Test Loss', fontsize=14)
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    # ax.legend(['Loss'], loc='lower right', fontsize=14)
    plt.show()
    
def plot_accuracy(epochs, acc_test):

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(epochs, acc_test)
    ax.set_title('Test Accuracy', fontsize=14)
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    # ax.legend(['Accuracy'], loc='lower right', fontsize=14)
    plt.show()

data = load_data('my_train_SGD_2.0.dat')
_, loss, _, acc, epochs = record_stats(data)

plot_loss(epochs, loss)
plot_accuracy(epochs, acc)
            
    
            
    

