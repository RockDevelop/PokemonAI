# PokemonAI - Machine Learning Final Project

## Description
The main file is pokemon.py. This is a Python script meant to be run with command line inputs to choose which model you wish to run.
You have the choice to run a decision tree model or a neural network model. The model will predict based on that stats and body type of a Pokemon what type it will be. The ouput will be the accuracy of the training followed by the accuracy of the testing. There are also visual aids to see how the model is performing. 
This program requires several python libraries to run. The required libraries are listed below.

## Required Libraries
* [Tensorflow](https://www.tensorflow.org/) - Backbone of the machine learning models
```
pip install tensorflow
```
* [Keras](https://keras.io/) - Used to create, train, and test the neural network model
```
pip install keras
```
* [Matplotlib](https://matplotlib.org/) - Used to plot and show the graphs related to the models
```
pip install -U matplotlib
```
* [Scikit-learn](https://scikit-learn.org/stable/) - Used to create, train, and test the decision tree model
```
pip install -U scikit-learn
```
* [Numpy](https://numpy.org/) - Used for data manipulation and ease of access to more complex data structres
```
pip install numpy
```
* [Wordcloud](https://pypi.org/project/wordcloud/) - Used for the visualization of the neural network predictions
```
pip install wordcloud
```
## How to run
You have two choices when running the model. You are able to choose if you want to run
the neural network model, or the decision tree model. The command line arguments are below.

* Neural Network
```
python pokemon.py -model nn
```
* Decision Tree
```
python pokemon.py -model dt
```
