
## Early stopping


The default method for improving generalization is called Early stopping. 

In this technique the available data is divided into three subsets. The general divison of the data set is as such :

Toatal Data Set => TRG(80%) + Test (20%)
TRG => Training (80%) + Validation (20%)

For example if the total data set is of 1000 samples, then TRG = 800 , Test =200 and

TRG is further divided into Training Set = 640 (80% of 800) , Validation Set = 160 (20% of 800)

The first subset is the Training set, which is used for computing the gradient and updating the network weights and biases. The second subset is the Validation set. The error on the validation set is monitored during the training process. The validation error normally decreases during the initial phase of training, as does the training set error. However, when the network begins to overfit the data, the error on the validation set typically begins to rise. When the validation error increases for a specified number of iterations , the training is stopped, and the weights and biases at the minimum of the validation error are retained. 

This process is clearly depicted in the below image which shows that the validation error increases as the number of epochs increases. Here for every 10 epochs the validation set is fed to the model and we note down the MSE. The Validation is always greater than or equal to the training error. If the Validation error increases at 3 continuous validation teast then wee stop the training beacause the model is being overfitted. At this point if we are satisfied with the training error then we can stop the training process.Test set is presented only once unlike the training set and the validation set. Test error is observed to be nearer to minimum validation error.
<img src="Images/image1.png" height="400" width="400"/>

The test set error is not used during training, but it is used to compare different models. It is also useful to plot the test set error during the training process. If the error in the test set reaches a minimum at a significantly different iteration number than the validation set error, this might indicate a poor division of the data set.


## Regularization

Regularization is a modification we make to the learning algorithm or the model architecture that reduces its generalisation error, possibly at the expense of increased training error. There are various ways of doing this, some of which include restriction on parameter values or adding terms to the objective function, etc.

### Weight Decay:
Weight decay is a regularization term that penalizes big weights.Because, if the weights are large then the sensitivity is lost. When the weight decay coefficient is big, the penalty for big weights is also big, when it is small weights can freely grow. The regularization parameter $\gamma$ determines how you trade off the original cost E with the large weights penalization. The loss function is given by the following equation:

$ E = \frac{1}{2}\displaystyle\sum_{i=1}^{N} (y_{id}-y_i)^2 + \gamma \displaystyle\sum_{i}\displaystyle\sum_{j}w_{ij}^2 $ where $\gamma \in [0,1] $. $\gamma$ tells how much importance has to be given for the regularization term.

Here the regularization term is $ \gamma \displaystyle\sum_{i}\displaystyle\sum_{j}w_{ij}^2 $ which effectively multiplies every weight by $(1-2\gamma\eta)$. 
 
 A drawback of weight decay was that we had to manually tweak the weight decay coefficient, which, if chosen wrongly, can lead the model to local minima by squashing the weight values too much. 
 
 
 ### Tangent Prop:
 Tangent propagation is a way of regularizing neural nets. It encourages the representation to be invariant by penalizing large changes in the representation when small transformations are applied to the inputs. The loss function for this is given by the following equation:

$ E=\frac{1}{2}\displaystyle\sum_{i=1}^{N} [(y_{id}-y_i)^2 + \mu\displaystyle\sum_{k=1}^{L}({\dfrac{\partial y_{id}}{\partial x_k}}-{\dfrac{\partial y_{i}}{\partial x_k}})^2]$ where $\mu \in [0,1]$ gives the relative importance of the regularization term.


The regularization term takes control over theamount of change in the classification output with the change in the input. When the  value of $y_{id}$ equals $y_i$ the 1st term of the loss function i.e., $(y_{id}-y_i)^2$ becomes zero but the regularization term is a non zero value. 

<img src="Images/image2.png" height="400" width="500"/>

The dotted line in the above image is the $Y_{est}$ which is the curve after learning.


##  Gradient Problems

### Vanishing Gradient
Vanishing Gradient Problem occurs when we try to train a Neural Network model using Gradient based optimization techniques.

When we do Back-propagation i.e moving backward in the Network and calculating gradients of loss(Error) with respect to the weights , the gradients tends to get smaller and smaller as we keep on moving backward in the Network. This means that the neurons in the Earlier layers learn very slowly as compared to the neurons in the later layers in the Hierarchy. The Earlier layers in the network are slowest to train.

In back propagation where Sigmoid activation function is used, we know that the weight updates are as follows

$\Delta w_{ij}=-\eta\delta_ix_j = -\eta(y_i-y_{id})y_i(1-y_i)x_j$

where $ y_i(1-y_i) $ is the sigmoid derivative. The maximum value of derivative will be 0.5 \* 0.5 =0.25. AT the next lower hidden layer it gets reducced to 0.25 \* 0.25=0.0625. The derivative pulls down the weight updates. This is the problem of Vanishing Gradients.


Earlier layers in the Network are important because they are responsible to learn and detecting the simple patterns and are actually the building blocks of our Network. Obviously , if they give improper and inaccurate results , then how can we expect the next layers and the complete Network to perform nicely and produce accurate results.

Hence because of Vanishing Gradient the Training process takes too long and the Prediction Accuracy of the Model will decrease.

There are two Layer dependent learning rate solutions for Vanishing Gradient problem:

1) $\eta_h = 4\eta$

2) RELU Activation function. The derivative of this function is given by

$ f'(x) =
  \begin{cases}
    1       & x>0 \\
    0  & x<=0
  \end{cases}
$

The vanishing gradient problem vanishes and the effectiveness of error remains in updating the weights.

### Exploding Gradient

 sometimes the gradient gets much larger in earlier layers! This is the exploding gradient problem, and it's not much better news than the vanishing gradient problem. More generally, it turns out that the gradient in deep neural networks is unstable, tending to either explode or vanish in earlier layers.
 
 If the weight values chosen are greater than 1 or activation functions are used whose derivatives can take on larger values then it results in exploding gradient problem. The most common approach is the so-called <b>“Gradient Clipping”</b>, but regularization can also help
