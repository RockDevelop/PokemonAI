# PokemonAI - Machine Learning Final Project

## File Description
`pokemon.py` - This is a Python script meant to be run with command line inputs to choose which model you wish to run.
You have the choice to run a decision tree model or a neural network model. The model will predict based on that stats and body type of a Pokemon what type it will be. The ouput will be the accuracy of the training followed by the accuracy of the testing. There are also visual aids to see how the model is performing.  

`data.csv` - This is a data set from [Kaggle](https://www.kaggle.com/datasets/alopez247/pokemon) that contains information on the first VI generations of pokemon games. The data is in the order below  
**Number**|**Name**|**Type 1**|**Type 2**|**Stats Total**|**HP**|**Attack**|**Defense**|**Sp Attack**|**Sp Defense**|**Speed**|**Generation**|isLegendary |**Color**|**hasGender**|**Percent Male**|**Egg Group 1**|**Egg Group 2**|**hasMegaEvolution**|**Height Meters**|**Weight Kilograms**|**Catch Rate**|**Body Type**
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:

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
