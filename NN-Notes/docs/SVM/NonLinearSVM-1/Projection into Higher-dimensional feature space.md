
## Projection into Higher-dimensional feature space

Let us consider the below non linear data set. We can see clearly that they are not linearly seperable. Consider the inner region to be -ve and the outer region to be +ve. Circle (non-linear) is the good solution vector for such dataset.


```python
from Figures import figure1
figure1.draw2D()
```


![png](Projection%20into%20Higher-dimensional%20feature%20space_files/Projection%20into%20Higher-dimensional%20feature%20space_1_0.png)


There is a different way to project this data. Generally there are many algorithms that reduce the dimensionality of the input data like PCA(Principle Component Analysis). But in this problem we will project the data into higher dimension. For example, in this case it takes the 2D dataset and projects it into higher dimension like 3D.

Let us consider ($x_1,x_2$) to be the original data point then it is changed to 3D as below

($x_1,x_2,x_3)=(x_1,x_2,x_1^2+x_2^2$). 
i.e., $\phi: \mathbb{R}^2 \mapsto \mathbb{R}^3$


Using this transformation, the points that are present nearer to the origin will pushed down and the points that are away from origin will be pulled up.

Below is the illustration that shows the higher dimensional projection of the dataset. It is observed that the plane seperates these points.


```python
figure1.draw3D()
```


![png](Projection%20into%20Higher-dimensional%20feature%20space_files/Projection%20into%20Higher-dimensional%20feature%20space_3_0.png)

