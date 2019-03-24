# NN-Notes

This repository will house the entire blog for the course taught by Dr. Raju S. Bapi, SCIS, UoH. Click [here](https://github.com/somanath08/NN-Notes/blob/master/contribution.md) to know more about contributing.

## TODO (There is more to complete than completed!)

### MLFNN

- To include a step-by-step process (preprocessing, network desgin, training and validation) of implementing a real world example. For example, Face Recognition, Stock Market Prediction, Bankruptcy Prediction, etc.

### RBF Network

- Theory (additive model), Algorithm, Implementation example

### Single hidden layer feedforward neural network (SLFN)

- This idea goes back to the original perceptron of Rosenblatt, where the input layer (Retina) is connected with random weights to the hidden layer (Association Cortex) and then onto the output layer (Motor Cortex) by learnable weights. Recently there are exciting new developments in these kind of random networks. For example, Extreme Learning Machine (ELM) and Random Vector Functional Link (RVFL) Neural Networks.
- Theory, Algorithm, Example of ELM
- Theory, Algorithm, Example of RVFL NN

### Limitations of Multilayer Perceptrons (Motivation to CNN)

- Vansihing gradient problem with increase in number of hidden layers of an MLP
  1. ReLU as a solution
  2. Layer-wise training as a strategy
- Inability to work with differing input sizes (images of different sizes, for example)
- Parameter explosion problem with standard MLP (with sigmoid or ReLU)
  1. Convolution and parameter sharing as a way of reducing the number of learnable parameters

### CNN

- Original Fukushima (1980) Neocognitron as the seed idea for CNN
- Building blocks of CNN (Padding, Stride, Multiple Filters, Conv over 3D, Detector, Pooling, Fully connected layer, etc)
- Standard examples: LeNet-5 (Sigmoid), ImageNet database, AlexNet (ReLU), VGG-19
- Vanishing gradient problem, again! Skip connections, ResNet
- 1x1 convolution, Inception (GoogLeNet)
- Real world example in CNN: use of packages such as Tensor Flow, etc, Computational requirements, use of pre-trained nets

### RNN

- feedback connection allows "memory" in the net
- RNNs hard to train: unfold to make them MLFFNN and use BPTT
- RNN implementation example: figure-of-eight generation, timeseries prediction, etc.

### LSTM

- Vanishing gradient problem (VGP), again with large unfolded nets!
- Motivation for LSTM that allows error propagation over long time intervals without VGP!
- Types of RNN and potential applications (figure from Karpathy's blog)
- LSTM building blocks: Input gate, cell state, forget gate, output gate, peephole connections, GRU
- Implementation examples

### SVM

- Implementation examples (using svmlib)

### Unsupervised Learning

- SOM applications
- Autoencoder (AE) for representation learning (Not supervised, but is it fully unsupervised?)
  1. encoder, decoder, tied weights
  2. undercomplete, overcomplete representations, denoising AE, contractive AE
  3. deep autoencoder

### Hopfield Nets

- Perspective from Memory, Autoassociation, Optimization
- Hand-worked example with state transition diagram, encoding and retrieval
- Optimization example of 8-queen problem, Travelling Sales person (TSP) problem

### Boltzmann Machines

- Hopfield net with hidden layer and stochastic activation function
- Simulated Annealing
- Contrastive divergence learning rule

### RBM and DBN

- Restricted Boltzmann Machine
- Deep Belief Nets

### Reinforcement Learning (RL)

- Classical (Pavlovian) versus Operant (Instrumental) conditioning
- Rescorla-Wagner model of conditioining
- Klopf's Drive-reinforcement model
- Sutton-Barto's Actor-Critic Model
- Value iteration, Policy iteration hand-coded examples
- TD model
- TD and Dopamine in Basal Ganglia
- Q-learning
- SARSA
- Model-based versus Model-free RL
- Deep RL
- Examples:
  1. Actor-Critic solving Cart-pole problem, mountain car problem
  2. TD-Gammon
  3. Alpha-Go
