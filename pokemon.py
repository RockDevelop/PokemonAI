''' 
Description:    Creating a machine learning model that will predict what type the pokemon is based
                on the given stats and body style.
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
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix

ROOT = os.path.dirname(os.path.abspath(__file__)) # Root directory of this code
TRAIN_SPLIT = 0.9

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

def decision_tree(data):
    # Extracting useful stats (hp, atk, def, spatk, spdef, speed)
    stats = data[:, 5:11].astype(np.int32)
    # Extracting all the types for each pokemon
    types = data[:, 2]
    # Extracting body style for each pokemon
    body_style = data[:, 22]

    _, uniquebodies = np.unique(body_style, return_inverse=True)
    
    # Combine stats and body_style into one input array
    stats = np.hstack((stats,uniquebodies[:, None]))
    
    # Creating a mapping of all the types to numbers
    # Then converting the types array to those numbers that way we can get an output that neural network will undestand
    _, uniquetypes = np.unique(types, return_inverse=True)
    maxt = np.max(uniquetypes) + 1

    # Spliting the data into a training and a testing set
    xtrain, ytrain, xtest, ytest = splitData(stats, uniquetypes, TRAIN_SPLIT, True)

    # Creating Decision Tree
    myTree = DecisionTreeClassifier(criterion='entropy')
    myTree = myTree.fit(xtrain, ytrain)

    # Test Decision Tree
    testPrediction = myTree.predict(xtest)
    trainPrediction = myTree.predict(xtrain)

    # Show confusion matrix for test data
    test = confusion_matrix(ytest, testPrediction)
    train = confusion_matrix(ytrain, trainPrediction)

    # Accounting for missing lables
    labels = np.unique(ytest)
    missing = []
    index = 0
    offset = 0
    while index < len(_):
        if index != labels[index - offset]:
            missing.append(index)
            offset += 1
        index += 1
    print(f'Test: {np.shape(test)}')
    print(f'Missing: {np.shape(missing)}')
    for element in missing:
        if element not in np.unique(testPrediction):
            test = np.insert(test, element, np.zeros(len(test[0])),0)
            test = np.insert(test, element, np.zeros(len(test)),1)

    # Shrinking testing confusion matrix
    testOutcome = []
    for row in range(len(test)):
        correct = 0
        error = 0
        for element in range(len(test[0])):
            if row == element:
                correct = test[row][element]
            else:
                error += test[row][element]
        testOutcome.append([correct, error])
    testOutcome = np.array(testOutcome)

    # Shrinking training confusion matrix
    trainOutcome = []
    for row in range(len(train)):
        correct = 0
        error = 0
        for element in range(len(train[0])):
            if row == element:
                correct = train[row][element]
            else:
                error += train[row][element]
        trainOutcome.append([correct, error])
    trainOutcome = np.array(trainOutcome)

    # Compare training and test accuracy
    testAccuracy = np.sum(testOutcome[:,0]) / np.sum(testOutcome)
    trainAccuracy = np.sum(trainOutcome[:,0]) / np.sum(trainOutcome)
    print("Training Accuracy: " + str(trainAccuracy))
    print("Testing Accuracy: " + str(testAccuracy))
   

    # Making Bar Graph for Data
    x = 0.5 + np.arange(len(_))
    correct = testOutcome[:,0]
    incorrect = testOutcome[:,1]

    fig, ax = plt.subplots()
    colors = ['#A6B91A', '#705746', '#6F35FC', '#F7D02C', '#D685AD', '#C22E28', '#EE8130', '#A98FF3', '#735797', '#7AC74C', '#96D9D6', '#A8A77A', '#A33EA1', '#F95587', '#B6A136', '#B7B7CE', '#6390F0']
    ax.bar(x, correct, width=1, align='edge', edgecolor='white', linewidth=0.7, color=colors)
    ax.bar(x, incorrect, width=1, bottom=correct, align='edge', edgecolor='white', linewidth=0.7, color='r')
    plt.xticks(np.arange(len(_)), _, color='black', rotation=45, fontsize='12', horizontalalignment='right')
    plt.xlabel('Pokemon Types', fontweight='bold', color = 'orange', fontsize='17', horizontalalignment='center')
    plt.ylabel('Number of Guesses Per Type', fontweight='bold', color = 'orange', fontsize='17', horizontalalignment='center')

    ylim = 0
    for element in testOutcome:
        temp = np.sum(element)
        if temp > ylim:
            ylim = temp
    ax.set(xlim=(0, len(_)+1), xticks=np.arange(0, len(_)), ylim=(0, ylim + 1), yticks=np.arange(0, len(_)))
    
    # Graphing Decision Tree (Which is a mistake)
    plt.figure()
    plot_tree(myTree, class_names = _, filled=True, rounded=True, fontsize=2)
    plt.show()


    
def neural_network(data):
    # Extracting useful stats (hp, atk, def, spatk, spdef, speed)
    stats = data[:, 5:11].astype(np.int32)
    # Extracting all the types for each pokemon
    types = data[:, 2]
    # Extracting body style for each pokemon
    body_style = data[:, 22]

    _, uniquebodies = np.unique(body_style, return_inverse=True)
    
    # Combine stats and body_style into one input array
    stats = np.hstack((stats,uniquebodies[:, None]))
    
    # Creating a mapping of all the types to numbers
    # Then converting the types array to those numbers that way we can get an output that neural network will undestand
    _, uniquetypes = np.unique(types, return_inverse=True)
    maxt = np.max(uniquetypes) + 1
    onehot = np.eye(maxt)[uniquetypes]

    # Spliting the data into a training and a testing set
    xtrain, ytrain, xtest, ytest = splitData(stats, onehot, TRAIN_SPLIT, True)

    # Creating Neural Network
    model = Sequential()
    model.add(Input(shape=(7,)))
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
    es = EarlyStopping(monitor='accuracy', mode='max', verbose=1, patience=30)

    # If you wanted to save the model this would allow you to do so
    # mcp_save = ModelCheckpoint('.saved_model.hdf5', save_best_only=True, monitor='accuracy', mode='max')
    reduce_lr_loss = ReduceLROnPlateau(monitor='accuracy', factor=0.1, patience=10, verbose=1, mode='max')
    # Train
    history = model.fit(xtrain, ytrain, epochs=2000, batch_size=10, verbose=1, validation_split=0.1, callbacks=[es, reduce_lr_loss])

    # Test
    metrics = model.evaluate(xtest, ytest, verbose=1)

    # y contains the guessed types and actual_index contains the correct indexes
    y = np.argmax(model.predict(xtest, verbose=0), axis=1)
    _, actual_index = np.where(ytest)

    # Creating a bar graph
    

    # Convert the guessed types from numbers into a string
    guessed_types = []
    for elem in y:
        guessed_types.append(np.unique(types)[elem])
    
    # Conver the actual types from one hot arrays into a string
    actual_types = []
    for elem in actual_index:
        actual_types.append(np.unique(types)[elem])

    # Convert the output into a single string instead of a list
    guessed_text = " ".join(guessed_types)
    actual_text = " ".join(actual_types)

    # Generate a word cloud for the guessed types   
    guessed_wordcloud = WordCloud(background_color='white').generate(guessed_text)
    plt.imshow(guessed_wordcloud, interpolation='bilinear')
    plt.axis(False)
    plt.title("Neural Network Testing Prediction")

    # Generate the word cloud for the actual types
    plt.figure()
    actual_wordcloud = WordCloud(background_color='white').generate(actual_text)
    plt.imshow(actual_wordcloud, interpolation='bilinear')
    plt.axis(False)
    plt.title("Neural Network Testing Actual")

    # Display the model
    plt.figure()
    plt.imshow(plt.imread('model_plot.png'))
    plt.axis(False)
    plt.title('Visual Representation of the Neural Network Layers')
    plt.show()
    

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
def splitData(stats, onehot, train_split, shuffle):
    # First get the number of cases that are going to be a part of our training set.
    # Then setup a list of potential indexes
    trainNum = int(len(stats) * train_split)
    indexes = np.arange(len(stats))

    # Shuffle the array when specified to achieve a varied training set
    if shuffle: 
        np.random.shuffle(indexes)

    # Create two arrays that contain the indexs for the training set and then the indexes for the test set
    training_indexes = indexes[0:trainNum]
    testing_indexes = indexes[trainNum:len(stats)]

    # Initialize empty lists
    xtrain = []
    ytrain = []
    xtest = []
    ytest = []
    
    # Loop over the indexes and append the NDarrays to the list
    for i in training_indexes:
        xtrain.append(stats[i])
        ytrain.append(onehot[i])
    for i in testing_indexes:
        xtest.append(stats[i])
        ytest.append(onehot[i])

    # Convert all the lists into NDarrays
    xtrain = np.array(xtrain)
    ytrain = np.array(ytrain)
    xtest = np.array(xtest)
    ytest = np.array(ytest)

    # Make sure there is at least 1 of

    # Return the NDarray
    return xtrain, ytrain, xtest, ytest

if __name__ == "__main__":
    main(parser.parse_args())