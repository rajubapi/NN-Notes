
## Kernels

For non linearly seperable data we can use Soft Margin SVM.

#### Training Phase
During the training phase the dual form $Q(\alpha)$ is given as

$Q(\alpha ) = \displaystyle\sum_{i=1}^{N} \alpha_i - \frac{1}{2}\displaystyle\sum_{i=1}^{N}\displaystyle\sum_{j=1}^{N} \alpha_i \alpha_j d_id_j\big[\phi(x_i).\phi(x_j)\big]  $

Subject to : $0\leq\alpha\leq C, \quad \forall i\in[1,N] \quad and \quad  \Sigma\alpha_id_i=0$


The solution weight vector($\tilde w^*$) and the bias($b^*$) are given as:

$\tilde w^* = \displaystyle\sum_{i=1}^{N}\alpha_id_i\phi(\tilde x_i)$
$b^* = 1-\big[w^*.\phi(\tilde x_i^*)\big]$

#### Testing Phase
In the testing phase $x_i^{test}$ is the inout for which class lable has to be determined and $d_{test}$ is the desired value of the corresponding test input.

$d_{test}$ can be given as the $Sign\big[w^*.\phi({x_i}^{test})+b^*\big]$. This means that if the value of $[w^*.\phi({x_i}^{test})+b^*]$ evaluates to a +ve value then the input belongs to one class and if it is -ve then it belongs to another class.

Substituting the value of $w^*$ in $d_{test}$ we get

$d_{test}=\big[\big(\displaystyle\sum_{i=1}^{N}\alpha_id_i\phi(\tilde x_i)\big).\phi({x_i}^{test})+b^*\big]$

If the input has 1000 dimensions then the dot product calculation of $\phi(\tilde x_i).\phi({x_i}^{test})$ will be tidious operation.


Few kernels like $\phi:\mathbb{R}^2\mapsto\mathbb{R}^3 $have a very nyc dot product. We don't need to find the dot product in higher dimensional space. Such kernels are called as **Mercer Kernel**. This dot product is equivalent to the dot product in original dimensional space.Therefore, we need not transform the input into higher dimension and then calculate the dot product, instead we can just perform the dot product in the lower dimension and then perform the transfrmation which saves the number of computations performed. But not all kernels satisfy this property

Let us consider one example:

$\phi(\tilde x)=\phi\bigg(\begin{bmatrix}x_1 \\ x_2\end{bmatrix}\bigg) = \begin{bmatrix}x_1^2\\ x_1x_2\\ x_2x_1\\ x_2^2\end{bmatrix}$ i.e., the Kernel converts 2D to 4D.

Let $K(x,y)=\phi(\tilde x).\phi(\tilde y)=\begin{bmatrix}x_1^2\\ x_1x_2\\ x_2x_1\\ x_2^2\end{bmatrix}.\begin{bmatrix}y_1^2\\ y_1y_2\\ y_2y_1\\ y_2^2\end{bmatrix}$ where k is an operator that performs dot product.

$=(x_1^2y_1^2)+2x_1x_2y_1y_2+x_2^2y_2^2$

$=(x_1y_1+x_2y_2)^2$

$=\begin{bmatrix}x_1 \\ x_2\end{bmatrix}\begin{bmatrix}y_1 \\ y_2\end{bmatrix}$

$k(\tilde x,\tilde y)=(\tilde x,\tilde y)^2$

This means that 2D input can be used to calculate the 4D dot product without actually converting it to 4D. Hence we observe that $\phi(x_i)$ need not be calculated.

Hence while calculating the dot products in $w^*$ and $b^*$ we don't need to convert them to higher dimension and calculate. We can just calculate the dot product in lower dimension.

### Computational Savings in Cubic Kernel
$K(X,Y)=(X,Y)^3$

$=(x_1y_1+x_2y_2)^3$

$\phi(x)=\phi\begin{pmatrix}\begin{bmatrix}x_1\\ x_2\end{bmatrix}\end{pmatrix}=\begin{bmatrix}x_1^3 \\ \sqrt {3}x_1^2x_2\\ \sqrt {3}x_1x_2^2 \\ x_2^3 \end{bmatrix}$ i.e., 2D $\rightarrow$ 4D

Computations for dot product:

Without kernel trick : 16 Multiplications and 3 additions.

With kernel trick : 4 Multiplications and 1 addition.

Consider another kernel that converts 3D to 10D like below

$\phi(\tilde x))=\phi\begin{pmatrix}\begin{bmatrix}x_1 \\ x_2\\ x_3\end{bmatrix}\end{pmatrix}= \begin{bmatrix}x_1^3 \\ x_2^3\\ x_3^3 \\
\sqrt{3}x_1^2x_2 \\ 
\sqrt{3}x_1^2x_3 \\ 
\sqrt{3}x_2^2x_3 \\ 
\sqrt{3}x_2^2x_1 \\ 
\sqrt{3}x_3^2x_1 \\ 
\sqrt{3}x_3^2x_2 \end{bmatrix}$

$K(X,Y) = (x_1y_1 + x_2y_2 +x_3y_3)^3$

Without kernel trick : 38 Multiplications and 9 additions.

With kernel trick : 5 Multiplications and 2 addition.

## Mercer Conditions

K(X,Y) should satisfy the below conditions inorder to be called as a Mercer Kernel

<ul>
    <li>Symmetric : $K(\tilde x,\tilde y)=K(\tilde y,\tilde x) $</li>  
    <li>Continuous</li>
    <li>Positive Semidefinite</li> 
    <li>$K(\tilde x,\tilde y)=\displaystyle\sum_{i=1}^{\infty}\lambda_i\phi_i(\tilde x).\phi_i(\tilde y) ; \lambda_i \gt 0$</li> 
</ul>

## Popular Kernels
<ul>
    <li>Ploynomial Kernel : $K_{\phi}(\tilde X,\tilde Y)=(1+XY)^p $</li>  
    <li>RBF (Radial Bassis Function) kernel or Gaussian Kernel: $K_{\gamma}(\tilde X,\tilde Y)=e^{-\frac{1}{2\sigma^2}\|X-Y\|^2}$ where $\sigma$ is the Standard deviation. This Kernel maps to infinite dimension space.</li>
    <li>Hyperbolic Tanget: $K_S(\tilde X,\tilde Y)=\tanh(\beta_0(\tilde X.\tilde Y)+\beta_1) $</li> 
</ul>
