
## Recoding XOR
<img src="Images/image1.gif" height="400" width="400"/>
A verbal description of the XOR problem is that the output must be turned on when either of the inputs is turned on, but not when both are turned on. Four inequalities must be satisfied for the perceptron to solve this problem:

$(0 * w1) + (0 * w2) <  \theta ==> 0 < \theta $

$(0 * w1) + (1 * w2) >  \theta ==> w2 > \theta $

$(1 * w1) + (0 * w2) >  \theta ==> w1 > \theta $

$(1 * w1) + (1 * w2) <  \theta ==> w1 + w2 < \theta $

 However, this is obviously not possible since both w1 and w2 would have to be greater than while their sum w1 + w2 is less than . There is an elegant geometric description of the types of problems that can be solved by a perceptron.

Consider three two-dimensional problems: AND, OR, and XOR. If we represent these problems geometrically (Figure 4), the binary inputs patterns form the vertices of a square. The input patterns 00 and 10 form the bottom left and top left corners of the square and the patterns 01 and 11 form the bottom right and top right corners of the square.

<img src="Images/image2.gif" height="300" width="300"/>
<img src="Images/image3.gif" height="150" width="150"/>

All problems which can be solved by separating the two classes with a hyperplane are called linearly separable. The XOR problem (as presented) is not a linearly separable problem. 

However, we can recode the XOR problem in three dimensions so that it becomes linearly separable.

<img src="Images/image4.gif" height="400" width="400"/>

In the three-dimensional version, the first two inputs are exactly the same as the original XOR and the third input is the AND of the first two. That is, input3 is only on when input1 and input2 are also on. By adding the appropriate extra feature, it is possible to separate the two classes with the resulting plane. Instead of recoding the input representation, another way to make a problem become linearly separable is to add an extra (hidden) layer between the inputs and the outputs. Given a sufficient number of hidden units, it is possible to recode any unsolvable problem into a linearly separable one.


### The Credit Assignment Problem in MultiLayer Networks
In the three-dimensional version of XOR network, we showed that a pre-determined extra input feature makes the XOR problem linearly separable. Consider when we add an extra layer to the network instead. The hidden layer provides a pool of units from which features (which help distinguish the inputs) can potentially develop. However, the outstanding question concerns how to learn those features. That is, how can a network develop an internal representation of a pattern?

Since there are no target activations for the hidden units, the perceptron learning rule does not extend to multilayer networks, The problem of how to train the hidden-unit weights is an acute problem of credit assignment. How can we determine the extent to which hidden-unit weights contribute to the error at the output, when there is not a direct error signal for these units. The <b>BackProp algorithm </b> provides a solution to this credit assignment problem. 

BackProp learning of XOR can be conceptualized as a two stage process. In the first (slow) stage, the network learns to recode the XOR problem so that it is easier to solve. In the slow stage of learning, the network is developing an internal representation of the XOR problem that is linearly separable. Once the XOR problem has been successfully recoded, learning proceeds quickly. 

There are different ways to recode the XOR problem so that it is linearly separable. The solution that the network finds depends on the precise values of the initial weights and biases. 


For example let us consider the neural network formed using below values of weights and biases.
<img src="Images/image7.png" height="400" width="400"/>
Using the weights and biases mentioned in the above figure we can implement XOR. Let us now look at all the inputs one by one.

#### Case 1: $x_1=0 , x_2 =0$

i) $w_1*x_1+w_2*x_2 = 1.0 *0 +-1.0*0 <0.5 $ => output at $h_1$ =0

ii) $w_1*x_1+w_2*x_2 = -1.0 *0 +1.0*0 <0.5 $ => output at $h_2$ =0

iii) $h_1*w_1+h_2*w_2 = 0 * 1.0 + 0*1.0 <0.5 $ => final output = 0

#### Case 2: $x_1=1 , x_2 =0$

i) $w_1*x_1+w_2*x_2 = 1.0 *1 +-1.0*0 >0.5 $ => output at $h_1$ =1

ii) $w_1*x_1+w_2*x_2 = -1.0 *1 +1.0*0 <0.5 $ => output at $h_2$ =0

iii) $h_1*w_1+h_2*w_2 = 1 * 1.0 + 0*1.0 >0.5 $ => final output = 1

#### Case 3: $x_1=0 , x_2 =1$

i) $w_1*x_1+w_2*x_2 = 1.0 *0 +-1.0*1 <0.5 $ => output at $h_1$ =0

ii) $w_1*x_1+w_2*x_2 = -1.0 *0 +1.0*1 >0.5 $ => output at $h_2$ =1

iii) $h_1*w_1+h_2*w_2 = 0 * 1.0 + 1*1.0 >0.5 $ => final output = 1

#### Case 4: $x_1=0 , x_2 =1$

i) $w_1*x_1+w_2*x_2 = 1.0 *1 +-1.0*1 <0.5 $ => output at $h_1$ =0

ii) $w_1*x_1+w_2*x_2 = -1.0 *1 +1.0*1 <0.5 $ => output at $h_2$ =0

iii) $h_1*w_1+h_2*w_2 = 0 * 1.0 + 0*1.0 >0.5 $ => final output = 0


Hence the XOR can be realised using these combinations of weights and biases. In the figure shown below the each of the lines represents a line in the weight space.
<img src="Images/image9.png" height="400" width="400"/>

Line I is $x_1-x_2>=0.5$ 

Line II is $x_2-x_1>=0.5$

Thus from the above figure as reference we can construct the below table


| Points | I - line | II - line | I or II [$y=h_1 \lor h_2$] |
| :---  | :---  | :---  | :--- |
| (0,0) , (1,1)  | 0  | 0 | 0 |
| (0,1) | 1  | 0 | 1 |
| (1,0) | 0  | 1 | 1 |


### Realising XOR with Back Propagation
One more way of realising XOR is by initialising the weights and bias to normalised random values and then training the neural network using the Back Propagation Algorithm. Below is the neural network that is implemented using 3-3-1 network model that is trained for 100000 epochs, $\eta $ = 0.2 and with Sigmoid as the activation function at both hidden layer as well as output layer. The output prediction after the training is as follows:


```python
from XORmlffnn import xor
mlp=xor.fit()
xor.printPrediction(mlp)
```

    Data	: 	Target	: 	Predicted
    -----------------------------------------
    [0 0] 	:	 0 	:	 0.008063361144289353
    [0 1] 	:	 1 	:	 0.9961567765433922
    [1 0] 	:	 1 	:	 0.9900804143061049
    [1 1] 	:	 0 	:	 0.00790297427689517


As the activation function used is Sigmoid the preddiction values approach to 0 and 1.
