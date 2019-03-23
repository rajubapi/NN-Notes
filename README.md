# NN-Notes

This repository will house the entire blog for the course taught by Dr. Raju S. Bapi, SCIS, UoH. Click [here](https://github.com/somanath08/NN-Notes/blob/master/contribution.md) to know more about contributing.

## TODO

## MLFNN
	* To include a step-by-step process (preprocessing, network desgin, training and validation) of implementing a real world example. For example, Face Recognition, Stock Market Prediction, Bankruptcy Prediction, etc. 

## RBF Network
	* Theory (additive model), Algorithm, Implementation example

## Single hidden layer feedforward neural network (SLFN)
	* This idea goes back to the original perceptron of Rosenblatt, where the input layer (Retina) is connected with random weights to the hidden layer (Association Cortex) and then onto the output layer (Motor Cortex) by learnable weights. Recently there are exciting new developments in these kind of random networks. For example, Extreme Learning Machine (ELM) and Random Vector Functional Link (RVFL) Neural Networks. 
	* Theory, Algorithm, Example of ELM
	* Theory, Algorithm, Example of RVFL NN

## Limitations of Multilayer Perceptrons
	* Vansihing gradient problem with increase in number of hidden layers of an MLP
		- ReLU as a solution
		- Layer-wise training as a strategy
	* Inability to work with differing input sizes (images of different sizes, for example)
	* Parameter explosion problem with standard MLP (with sigmoid or ReLU)
		- Convolution and parameter sharing as a way of reducing the number of learnable parameters

## CNN
	* Building blocks of CNN (Padding, Filters, Detector, 1x1 filter, etc)
* Okay lot of work

## Hyperlink to notes

### Perceptron

* [Linear Decision Boundary](https://github.com/somanath08/NN-Notes/blob/Perceptron/Perceptron-1/Linear-decision-boundary.ipynb) 

* [Sketchig Feasible Region](https://github.com/somanath08/NN-Notes/blob/Perceptron/Perceptron-2/Sketching_Feasibility_Region.ipynb)

* [Convergence Theroem](https://github.com/somanath08/NN-Notes/blob/Perceptron/Perceptron-3/Convergence.ipynb)

* [Loss functions](https://github.com/somanath08/NN-Notes/blob/Perceptron/Perceptron-4/loss-function.ipynb)

* [Bias vs No Bias](https://github.com/somanath08/NN-Notes/blob/Perceptron/Perceptron-5/Bais-vs-No-Bais.ipynb)

* [Activation functions](https://github.com/somanath08/NN-Notes/blob/Perceptron/Perceptron-6/Activation.ipynb)

### MLFFNN

* [Sigmoid: Influence of Bias and Weight](https://github.com/somanath08/NN-Notes/blob/k/sigmoid/MLFFNN-1/Sigmoid_Function.ipynb)

* [Function approximation](https://github.com/somanath08/NN-Notes/blob/s/function-approximation/MLFFNN-2/Function-approximation-using-sigmoid.ipynb)

* [Overfitting](https://github.com/somanath08/NN-Notes/blob/k/overfitting/MLFFNN-3/Overfitting.ipynb)

* [Non-linear classification](https://github.com/somanath08/NN-Notes/blob/s/non-linear-classifier/MLFFNN-4/Non-linear-classification-examples.ipynb)

* [Recoding in XOR](https://github.com/somanath08/NN-Notes/blob/k/xor/MLFFNN-5/Recoding%20XOR.ipynb)

* [Derivations of Backprop Equations](https://github.com/somanath08/NN-Notes/blob/s/derivations/MLFFNN-6/Derivations.ipynb)

* Practical Issues
    * [Part 1](https://github.com/somanath08/NN-Notes/blob/s/practical-issues-1/MLFFNN-7_Part1/Practical-Issues-Part-1.ipynb)

    * [Part 2](https://github.com/somanath08/NN-Notes/blob/k/practical-issues-2/MLFFNN-7/Practical%20Issues.ipynb)
