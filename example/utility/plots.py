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

    plt.plot(epochs, losses_test)
    plt.title('Test Loss')
    plt.legend()
    plt.show()
    
def plot_accuracy(epochs, acc_test):

    plt.plot(epochs, acc_test)
    plt.title('Test Accuracy')
    plt.legend()
    plt.show()


data = load_data('my_train_SGD_2.0.dat')
_, loss, _, acc, epochs = record_stats(data)

plot_loss(epochs, loss)
plot_accuracy(epochs, acc)
            
    
            
    

