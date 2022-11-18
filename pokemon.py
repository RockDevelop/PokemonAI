''' 
Description:    Creating an AI that will predict what type the pokemon is based on the given stats.
Authors:        Isaac Garay and Riley Peters
Links:          https://www.kaggle.com/datasets/alopez247/pokemon
                https://bulbapedia.bulbagarden.net/wiki/Stat
'''
import math
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import SGD, Adam
from keras.callbacks import Callback, EarlyStopping
import keras.utils
import pdb

def main():
    # Importing data
    # Currently, the data is all strings due to the data having some text fields.
    # We could modify the data table in the future if we want instead of converting a string to a value if we want to.
    ROOT = os.path.dirname(os.path.abspath(__file__))
    data = np.loadtxt(os.path.expanduser(os.path.join(ROOT, 'data.csv')), dtype=str, delimiter=",")
    
    # Extracting useful stats (hp, atk, def, spatk, spdef, speed)
    stats = data[:, 5:11].astype(np.int32)
    # Extracting all the types for each pokemon
    types = data[:, 2]
    
    # Creating a mapping of all the types to numbers
    # Then converting the types array to those numbers that way we can get an output that neural network will undestand
    _, uniquetypes = np.unique(types, return_inverse=True)
    maxt = np.max(uniquetypes) + 1
    onehot = np.eye(maxt)[uniquetypes]
    # typemapping = {}
    # for i in range(18):
    #     typemapping[uniquetypes[i]] = i
    
    # for i, element in enumerate(types):
    #     types[i] = int(typemapping.get(element))
    # types = types.astype(np.int32)
    

    # # Flipping the dictionary so when we get our values back, we can easily covert them to the associatedtypes pokemon type
    # typemapping = {v: k for k, v in typemapping.items()}

    # Creating Neural Network
    model = Sequential()
    model.add(Input(shape=(6,)))
    model.add(Dense(units=100, activation='swish', name='hidden1'))
    model.add(Dense(units=100, activation='swish', name='hidden2'))
    model.add(Dense(units=50, activation='swish', name='hidden3'))
    model.add(Dense(units=50, activation='swish', name='hidden4'))
    model.add(Dense(units=18, activation='softmax', name='output')) #softmax good when you have a neuron for each output
    model.summary()

    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    # Create an EarlyStopping callback
    # Stops the model if there is no improvement after 100 epochs
    es = EarlyStopping(monitor='accuracy', mode='max', verbose=1, patience=100)
    # Train
    history = model.fit(stats, onehot, epochs=2000, batch_size=10, verbose=1, callbacks=es)

    # Test
    metrics = model.evaluate(stats, onehot , verbose=0)

    # Displaying Data


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

if __name__ == "__main__":
    main()