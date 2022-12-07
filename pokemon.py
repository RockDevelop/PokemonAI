''' 
Description:    Creating an AI that will predict what type the pokemon is based on the given stats.
Authors:        Isaac Garay and Riley Peters
Links:          https://www.kaggle.com/datasets/alopez247/pokemon
                https://bulbapedia.bulbagarden.net/wiki/Stat
                https://www.graphviz.org/download/
'''
import h5py
import math
import sys
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import SGD, Adam
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils.vis_utils import plot_model
from sklearn.tree import DecisionTreeClassifier, plot_tree
import pdb
ROOT = os.path.dirname(os.path.abspath(__file__)) # Root directory of this code

parser = argparse.ArgumentParser(description="Train a neural network or decision tree to classify which type a pokemon is based on their stats")
parser.add_argument('-model', '--model', help='chooses to train either the neural network (nn) or the decision tree (dt) model - Example: python .\pokemon.py -model nn')

def main(args):
    # Importing data
    # Currently, the data is all strings due to the data having some text fields.
    # We could modify the data table in the future if we want instead of converting a string to a value if we want to.
    data = np.loadtxt(os.path.expanduser(os.path.join(ROOT, 'data.csv')), dtype=str, delimiter=",")
    
    # Extracting the modelChoice and running the correct function
    modelChoice = args.model
    if modelChoice == 'nn':
        neural_network(data)
    elif modelChoice == 'dt':
        decision_tree(data)
    else:
        print("Please run the model with either:\n    -model nn\n    -model dt")
        return
    # trainOnehot = onehot[0:len(train)]
    # testOnehot = onehot[len(train):len(onehot)]
    # typemapping = {}
    # for i in range(18):
    #     typemapping[uniquetypes[i]] = i
    
    # for i, element in enumerate(types):
    #     types[i] = int(typemapping.get(element))
    # types = types.astype(np.int32)
    

    # # Flipping the dictionary so when we get our values back, we can easily covert them to the associatedtypes pokemon type
    # typemapping = {v: k for k, v in typemapping.items()}
def decision_tree(stats, onehot):
    pass
    
def neural_network(data):
    # Extracting useful stats (hp, atk, def, spatk, spdef, speed)
    stats = data[:, 5:11].astype(np.int32)
    # Extracting all the types for each pokemon
    types = data[:, 2]
    # Spliting the data into a training and a testing set
    # train, test = splitData(stats, 0.8, False)
    
    # Creating a mapping of all the types to numbers
    # Then converting the types array to those numbers that way we can get an output that neural network will undestand
    _, uniquetypes = np.unique(types, return_inverse=True)
    maxt = np.max(uniquetypes) + 1
    onehot = np.eye(maxt)[uniquetypes]

    # Creating Neural Network
    model = Sequential()
    model.add(Input(shape=(6,)))
    model.add(Dense(units=100, activation='swish', name='hidden1')) 
    model.add(Dense(units=50, activation='swish', name='hidden2'))
    model.add(Dense(units=50, activation='swish', name='hidden3'))
    model.add(Dense(units=25, activation='swish', name='hidden4'))
    model.add(Dense(units=18, activation='softmax', name='output')) #softmax good when you have a neuron for each output
    model.summary()

    """This is strictly for generating the .png file that shows what our model looks like
        Uncomment the line below if you wish to update the .png if you changed the layout
        Of the model. Note: You have to have pydot installed AND graphviz installed
        The graphviz download link is in the Links section at the top of the file"""
    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    # Create an EarlyStopping callback
    # Stops the model if there is no improvement after some amount of epochs
    es = EarlyStopping(monitor='accuracy', mode='max', verbose=1, patience=40)
    mcp_save = ModelCheckpoint('.saved_model.hdf5', save_best_only=True, monitor='accuracy', mode='max')
    reduce_lr_loss = ReduceLROnPlateau(monitor='accuracy', factor=0.1, patience=15, verbose=1, mode='max')
    # Train
    history = model.fit(stats, onehot, epochs=2000, batch_size=10, verbose=1, validation_split=0.15, callbacks=[es, mcp_save, reduce_lr_loss])

    # Test
    metrics = model.evaluate(stats, onehot, verbose=1)

    # Display the model
    plt.imshow(plt.imread('model_plot.png'))
    plt.axis(False)
    plt.title('Visual Representation of the Neural Network Layers')
    plt.show()

    # Displaying Data
    # prediction = model.predict(test)
    pdb.set_trace()

'''Desciption: Calculates the HP stat for the pokemon
Inputs: base stat, iv, ev, level, gen
Output: HP stat as an int'''
def hp(base, iv, ev, level, gen):
    if gen >= 3:
        return math.floor(((2 * base + iv + math.floor(ev/4)) * level)/100) + level + 10
    else:
        return math.floor((((base + iv) * 2 + math.floor(math.ceil(math.sqrt(ev))/4)) * level)/100) + level + 10

'''Desciption: Calculates any stat for the pokemon other than HP
Inputs: base stat, iv, ev, level, gen, nature (2: benifits stat, 1: netural, 0: hinders stat)
Output: Stat as an int'''
def stat(base, iv, ev, level, gen, nature):
    if gen >= 3:
        if nature == 2: nature = 1.1
        elif nature == 1: nature = 1
        else: nature = 0.9

        return math.floor((math.floor(((2 * base + iv + math.floor(ev/4)) * level) / 100) + 5) * nature)
    else:
        return math.floor((((base + iv) * 2 + math.floor(math.ceil(math.sqrt(ev))/4)) * level)/100) + 5

'''Desciption: Splits the given data set into a training and testing set given a set percentage 
Inputs: data, training split percentage (as a decimal), shuffle boolean (whether or not the indexs it picks are random or just in order)
Output: An array for all the training indexes and an array for all the testing indexes'''
def splitData(data, train_split = 0.8, shuffle = True):
    # First get the number of cases that are going to be a part of our training set.
    # Then setup a list of potential indexes
    trainNum = int(len(data) * train_split)
    indexes = np.arange(len(data))

    # Shuffle the array when specified to achieve a varied training set
    if shuffle: 
        np.random.shuffle(indexes)

    # Create two arrays that contain the indexs for the training set and then the indexes for the test set
    training_indexes = indexes[0:trainNum]
    testing_indexes = indexes[trainNum:len(data)]

    trainingSet = np.empty((1,6))
    testingSet = np.empty((1,6))
    for element in training_indexes:
        trainingSet = np.append(trainingSet, data[element])
    for element in testing_indexes:
        testingSet = np.append(testingSet, data[element])

    pdb.set_trace()
    return training_indexes, testing_indexes

if __name__ == "__main__":
    main(parser.parse_args())