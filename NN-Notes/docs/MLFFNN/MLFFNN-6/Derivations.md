
# Derivation of Back Propagation equations

Before we look into the derivations we list the loss funcions and the activation functions below:



## Loss functions

A loss function is used to map a *cost* associated with an event. An optimization problem seeks to minimize the cost or loss.

Some common loss functions are:

### Mean squared error
$$
\text{E} =\sum_{i=1}^{N} \frac{1}{2} (y_{i} - y_{d})^2
$$
Here $y_i$ is the actual output and $y_d$ is the desired output

### Cross-entropy loss
#### a) For Sigmoid
$$
\text{E} = -\bigg\{\sum_{i=1}^N y_{id}\log y_i + (1 - y_{id})\log(1 - y_{i})\bigg\}
$$
Here $y_i$ is the actual output and $y_{id}$ is the desired output
#### b) For Softmax
$$
\text{E} = -\bigg\{\sum_{i=1}^N y_{id}\log y_i \bigg\}
$$
Here $y_i$ is the actual output and $y_{id}$ is the desired output


### L1 loss function
$$
\sum_{i=1}^{N} \bigg\lvert y_{i} - y_{d} \bigg\rvert
$$
Here $y_i$ is the actual output and $y_d$ is the desired output

## Activation functions

![Activation functions](Images/activations.png)

## Derivation of update equation using Mean-Squared-error

$$
\text{E} =\sum_{i=1}^{N} \frac{1}{2} (y_{i} - y_{d})^2
$$

We know,

$
\Delta w_{ij} = -\eta \frac{\partial \text{E}}{\partial w_{ij}}
$

$
\frac{\partial E}{\partial w_{ij}} = \frac{1}{2} * 2 * (y_i - y_{id}) * \frac{\partial (y_i - y_{id})}{\partial w_{ij}}
$

$
= (y_i - y_{id}) \frac{\partial y_i}{\partial w_{ij}}
$

$
\because \frac{\partial y_i}{\partial w_{ij}} = \frac{\partial y_i}{\partial z_{i}} \frac{\partial z_i}{\partial w_{ij}}
$

Here $z_i$ is the net input to the node.

$
\frac{\partial E}{\partial w_{ij}} = (y_i - y_{id}) \frac{\partial y_i}{\partial z_{i}} \frac{\partial z_i}{\partial w_{ij}}....(1)
$

Here,

$
z_i = \sum_{j=1}^N w_{ij}x_{j}
$

Where $x_{j}$ is the output of the jth hidden node.

$
\therefore \frac{\partial z_i}{\partial w_{ij}} = x_j
$

Substituting back in (1)

$
\frac{\partial E}{\partial w_{ij}} = (y_i - y_{id}) \frac{\partial y_i}{\partial z_{i}} x_j....(2)
$

The value of the partial derivative $\frac{\partial y_i}{\partial z_{i}}$ depends on the activation function used.


### For Sigmoid

$
\frac{\partial y_i}{\partial z_{i}} = y_i(1 - y_i)
$

Therefore, from (2)

$
\frac{\partial E}{\partial w_{ij}} = (y_i - y_{id}) y_i(1 - y_i) x_j
$

$
\delta_i = (y_i - y_{id}) y_i(1 - y_i)
$

$
\Delta w_{ij} = -\eta \delta_i x_j
$


### For tanh

$
\frac{\partial y_i}{\partial z_{i}} = (1 - y_i^2)
$

Therefore, from (2)

$
\frac{\partial E}{\partial w_{ij}} = (y_i - y_{id}) (1 - y_i^2) x_j
$

$
\delta_i = (y_i - y_{id}) (1 - y_i^2)
$

$
\Delta w_{ij} = -\eta \delta_i x_j
$


### For ReLU

$
\frac{\partial y_i}{\partial z_{i}} = 
    \begin{cases}
        0 \ if \ x \lt 0 \\
        1 \ if \ x \ge 0
    \end{cases}
$

Therefore, from (2)

$
\frac{\partial E}{\partial w_{ij}} = 
    \begin{cases}
        0 \ if \ x \lt 0 \\
        (y_i - y_{id}) x_j \ if \ x \ge 0
    \end{cases}
$

$
\delta_i = 
    \begin{cases}
        0 \ if \ x \lt 0 \\
        (y_i - y_{id}) \ if \ x \ge 0
    \end{cases}
$

$
\Delta w_{ij} = -\eta \delta_i x_j
$

### For Linear

$
\frac{\partial y_i}{\partial z_{i}} = 1
$

Therefore, from (2)

$
\frac{\partial E}{\partial w_{ij}} = (y_i - y_{id}) x_j
$

$
\delta_i = (y_i - y_{id})
$

$
\Delta w_{ij} = -\eta \delta_i x_j
$




## Derivation of update equation using Cross-entropy for Sigmoid

$$
\text{E} = -\bigg\{\sum_{i=1}^N y_{id}\log y_i + (1 - y_{id})\log(1 - y_{i})\bigg\}
$$

We know,

$
\Delta w_{ij} = -\eta \frac{\partial \text{E}}{\partial w_{ij}}
$

$
\frac{\partial E}{\partial w_{ij}} = -\bigg[\frac{y_{id}}{y_i} + \big(\frac{1-y_{id}}{1-y_i}\big)\bigg] \frac{\partial y_i}{\partial w_{ij}}
$

$
= \frac{(y_i - y_{id})}{y_i(1 - y_i)} \frac{\partial y_i}{\partial w_{ij}}
$

$
\because \frac{\partial y_i}{\partial w_{ij}} = \frac{\partial y_i}{\partial z_{i}} \frac{\partial z_i}{\partial w_{ij}}
$

Here $z_i$ is the net input to the node.

$
\frac{\partial E}{\partial w_{ij}} = \frac{(y_i - y_{id})}{y_i(1 - y_i)} \frac{\partial y_i}{\partial z_{i}} \frac{\partial z_i}{\partial w_{ij}}....(1)
$

Here,

$
z_i = \sum_{j=1}^N w_{ij}x_{j}
$

Where $x_{j}$ is the output of the jth hidden node.

$
\therefore \frac{\partial z_i}{\partial w_{ij}} = x_j
$

Substituting back in (1)

$
\frac{\partial E}{\partial w_{ij}} = \frac{(y_i - y_{id})}{y_i(1 - y_i)} \frac{\partial y_i}{\partial z_{i}} x_j....(2)
$

**Only sigmoid activation function is used in conjunction with the Cross-entropy loss function.**

### For Sigmoid

$
\frac{\partial y_i}{\partial z_{i}} = y_i(1 - y_i)
$

Therefore, from (2)

$
\frac{\partial E}{\partial w_{ij}} = \frac{(y_i - y_{id})}{y_i(1 - y_i)} y_i(1 - y_i) x_j
$

$
= \frac{\partial E}{\partial w_{ij}} = (y_i - y_{id}) x_j
$

$
\delta_i = (y_i - y_{id})
$

$
\Delta w_{ij} = -\eta \delta_i x_j
$


## Derivation of update equation using Cross-entropy for Softmax

$$
\text{E} = -\bigg\{\sum_{i=1}^N y_{id}\log y_i \bigg\}
$$

We know,

$
\Delta w_{ij} = -\eta \frac{\partial \text{E}}{\partial w_{ij}}
$

$
\frac{\partial E}{\partial w_{ij}} = -\bigg[\frac{y_{id}}{y_i}\bigg] \frac{\partial y_i}{\partial w_{ij}}
$



$
\because \frac{\partial y_i}{\partial w_{ij}} = \frac{\partial y_i}{\partial z_{i}} \frac{\partial z_i}{\partial w_{ij}}
$

Here $z_i$ is the net input to the node.

$
\frac{\partial E}{\partial w_{ij}} = -\bigg[\frac{y_{id}}{y_i}\bigg]\frac{\partial y_i}{\partial z_{i}} \frac{\partial z_i}{\partial w_{ij}}....(1)
$

Here,

$
z_i = \sum_{j=1}^N w_{ij}x_{j}
$

Where $x_{j}$ is the output of the jth hidden node.

$
\therefore \frac{\partial z_i}{\partial w_{ij}} = x_j
$

Substituting back in (1)

$
\frac{\partial E}{\partial w_{ij}} =-\bigg[\frac{y_{id}}{y_i}\bigg]\frac{\partial y_i}{\partial z_{i}} x_j....(2)
$

$\frac{\partial E}{\partial w_{ij}} =\frac{\partial E}{\partial y_i}\frac{\partial y_i}{\partial z_i}\frac{\partial z_i}{\partial w_{ij}} = \frac{\partial E}{\partial z_i}\frac{\partial z_i}{\partial w_{ij}}....(2)$

We know that Softmax is given by $y_i=\frac{e_{z_i}}{\sum_{j=1}^{N}e_{z_j}}$

$
\frac{\partial y_i}{\partial z_{j}} = 
    \begin{cases}
        y_i(1-y_j) \ if \  i= j \\
        -y_iy_j \ if \  i\neq j
    \end{cases}
$

$
\frac{\partial E}{\partial z_i} =
\begin{cases}
        -\bigg[\frac{y_{id}}{y_i}\bigg]y_i(1-y_j) \ if \  i= j \\
        -\sum_{j\neq i}\bigg(\bigg[\frac{y_{jd}}{y_j}\bigg](-y_iy_j)\bigg) \ if \  i \neq j 
    \end{cases}$

$
\frac{\partial E}{\partial z_i} =
\begin{cases}
        -y_{id}(1-y_j) \ if \  i= j \\
        \sum_{j\neq i}\bigg(y_{jd}y_i\bigg)\ if \  i \neq j 
    \end{cases}$


$
\frac{\partial E}{\partial z_i} =
        -y_{id}(1-y_j) +
        y_i\sum_{j\neq i}y_{jd}
$



$
\frac{\partial E}{\partial z_i} =
        -y_{id}+y_{id}y_i+
        y_i\sum_{j\neq i}y_{jd}
$

$
\frac{\partial E}{\partial z_i} =
        -y_{id}+y_i\bigg(y_{id}+\sum_{j\neq i}y_{jd}\bigg)
$


$\frac{\partial E}{\partial z_i} =[y_i-y_{id}] \ \bigg(\because \sum_{k}y_k=1\bigg)$

We know that,
$\frac{\partial E}{\partial w_{ij}} = \frac{\partial E}{\partial z_i}\frac{\partial z_i}{\partial w_{ij}} =(y_i-y_{id}) x_j$

$
\delta_i = (y_i - y_{id})
$

$
\Delta w_{ij} = -\eta \delta_i x_j
$



## Derivation of update equation using L1 loss

$$
\text{E} = \sum_{i=1}^{N} \bigg\lvert y_{i} - y_{d} \bigg\rvert
$$

We know,

$
\Delta w_{ij} = -\eta \frac{\partial \text{E}}{\partial w_{ij}}
$

$
\frac{\partial E}{\partial w_{ij}} = 
    \begin{cases}
        \frac{\partial y_i}{\partial w_{ij}} \ if \ y_i \gt y_d \\
        -\frac{\partial y_i}{\partial w_{ij}} \ if \ y_i \lt y_d
    \end{cases}
$

$
\because \frac{\partial y_i}{\partial w_{ij}} = \frac{\partial y_i}{\partial z_{i}} \frac{\partial z_i}{\partial w_{ij}}
$

Here $z_i$ is the net input to the node.

$
\frac{\partial E}{\partial w_{ij}} = 
    \begin{cases}
        \frac{\partial y_i}{\partial z_{i}} \frac{\partial z_i}{\partial w_{ij}} \ if \ y_i \gt y_d \\
        -\frac{\partial y_i}{\partial z_{i}} \frac{\partial z_i}{\partial w_{ij}} \ if \ y_i \lt y_d
    \end{cases}....(1)
$

Here,

$
z_i = \sum_{j=1}^N w_{ij}x_{j}
$

Where $x_{j}$ is the output of the jth hidden node.

$
\therefore \frac{\partial z_i}{\partial w_{ij}} = x_j
$

Substituting back in (1)


$
\frac{\partial E}{\partial w_{ij}} = 
    \begin{cases}
        \frac{\partial y_i}{\partial z_{i}} x_j \ if \ y_i \gt y_d \\
        -\frac{\partial y_i}{\partial z_{i}} x_j \ if \ y_i \lt y_d
    \end{cases}....(2)
$


The value of the partial derivative $\frac{\partial y_i}{\partial z_{i}}$ depends on the activation function used.


### For Sigmoid

$
\frac{\partial y_i}{\partial z_{i}} = y_i(1 - y_i)
$

Therefore, from (2)

$
\frac{\partial E}{\partial w_{ij}} = 
    \begin{cases}
        y_i(1 - y_i) x_j \ if \ y_i \gt y_d \\
        -y_i(1 - y_i) x_j \ if \ y_i \lt y_d
    \end{cases}
$

$
\delta_i = 
    \begin{cases}
        y_i(1 - y_i) \ if \ y_i \gt y_d \\
        -y_i(1 - y_i) \ if \ y_i \lt y_d
    \end{cases}
$

$
\Delta w_{ij} = -\eta \delta_i x_j
$


### For tanh

$
\frac{\partial y_i}{\partial z_{i}} = (1 - y_i^2)
$

Therefore, from (2)

$
\frac{\partial E}{\partial w_{ij}} = 
    \begin{cases}
        (1 - y_i^2) x_j \ if \ y_i \gt y_d \\
        -(1 - y_i^2) x_j \ if \ y_i \lt y_d
    \end{cases}
$

$
\delta_i = 
    \begin{cases}
        (1 - y_i^2) \ if \ y_i \gt y_d \\
        -(1 - y_i^2) \ if \ y_i \lt y_d
    \end{cases}
$

$
\Delta w_{ij} = -\eta \delta_i x_j
$


### For ReLU

$
\frac{\partial y_i}{\partial z_{i}} = 
    \begin{cases}
        0 \ if \ x \lt 0 \\
        1 \ if \ x \ge 0
    \end{cases}
$

Therefore, from (2)

$
\frac{\partial E}{\partial w_{ij}} = 
    \begin{cases}
        0 \ if \ x \lt 0 \\
        x_j \ if \ x \gt 0
    \end{cases}
$

$
\delta_i = 
    \begin{cases}
        0 \ if \ x \lt 0 \\
        1 \ if \ x \gt 0
    \end{cases}
$

$
\Delta w_{ij} = -\eta \delta_i x_j
$

### For Linear

$
\frac{\partial y_i}{\partial z_{i}} = 1
$

Therefore, from (2)

$
\frac{\partial E}{\partial w_{ij}} = x_j
$

$
\delta_i = 1
$

$
\Delta w_{ij} = -\eta \delta_i x_j
$



