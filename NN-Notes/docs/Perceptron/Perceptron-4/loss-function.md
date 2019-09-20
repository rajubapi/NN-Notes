
## Loss Function
The main use of loss function is to measure the difference or inconsistency between the predicted value ($ \begin{align}\hat{y}\end {align}$) and actual label ($ \begin{align}y\end {align}$). The robustness of the neural network increases with the decrease in the value of loss function.The number of misclassified patterns contribute to the change in inupt parameters. Few of the loss functions are listed below:
<ul>
    <li>Sum Squared Error</li>
    <li>Mean Squared Error</li>
    <li>Cross Entropy</li>
    <li>Minkowsky Error</li>
    <li>Regularization</li>
</ul>
Now let us look at Sum Squared Error, Mean Squared Error and Cross Entropy in detail

### Sum Squared Error
Squared error is given by the following formula

$ \begin{align} E = \sum_{p=1}^{P} (y_d^p-y^p)^2 \text{ where } p \text{ is the pattern number and } P \text{ is the total number of patterns} \end {align} $ 

The difference $ \begin{align} (y_d^p-y^p) \end {align} $ is raised to the power 2 because we do not care about the sign.


As we can now observe that only the patterns that are misclassified give a non zero loss value where as the pefectly classified pattern gives a zero loss value.

### Mean Squared Error ( MSE )
It is the mean of the sum squared error i.e., the the sum squared error divided by the total number of patterns gives the Mean Squared error. 
$ \begin{align} E = \frac{1}{P}\sum_{p=1}^{P} (y_d^p-y^p)^2 \text{ where } p \text{ is the pattern number and } P \text{ is the total number of patterns} \end {align} $ 



### Cross Entropy
Cross Entropy is commonly-used in binary classification (labels are assumed to take values 0 or 1) as a loss function (For multi-classification, use Multi-class Cross Entropy), which is computed by 

$ \begin{align} E = -\frac{1}{N}\sum_{i=1}^{N}[ y_{id}\log (y_i) + (1-y_{id})\log (1-y_i)] \end {align} $ 

The above function is generally used in conjunction with SIgmoid Activation function. This loss function maximizes conditional log likelihood of the training data. And more over the correlation between the estimated $ \begin{align}y_i \end {align} $ and the desired $ \begin{align} y_{id} \end {align} $ are minimised.


There is one more Cross Entropy functon that is used in conjunction with Softmax activation function.

$ \begin{align} E = -\frac{1}{N}\sum_{i=1}^{N}[ y_{id}\log (y_i) ]\quad\text{ where Softmax is given by } y = \frac{{e}^{z_i}}{\sum_{j=1}^{N}{e}^{z_j}}\end {align} $ 

