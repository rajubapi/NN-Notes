
## Cross Validation

Let us start with the question **"What is Cross Validation?"**

Cross-validation is a technique in which we train our model using the subset of the data-set and then evaluate using the complementary subset of the data-set.

In the beow illustration **k-fold Cross Validation** is used. In k-fold cross validation  we split the data-set into k number of subsets(known as folds) then we perform training on the all the subsets but leave one(k-1) subset for the evaluation of the trained model. In this method, we iterate k times with a different subset reserved for testing purpose each time.

<img src="Images/cv.png" height="400" width="400"/>

In each iteration we calculate the mean error which is called as the **Mean cross validation error**. This process is repeated for each of the models and the model which gives least Mean cross validation error is taken into consideration.

Here we use various SVM classifiers with different **C**(0.001,0.05,1,10,50) and **kernel**(linear,rbf) values and perform the k-fold cross validation (on 2D dataset with some noise) inorder to find out the best classifier with appropriate C and kernel.


```python
from Figures import figure1
figure1.performCVandgetResult()
```


![png](Cross%20Validation_files/Cross%20Validation_1_0.png)


    
    Kernel	C	Mean Cross Validation Error
    ---------------------------------------
    linear 	 0.001 	 0.41015625
    rbf 	 0.001 	 0.39990234375
    linear 	 0.05 	 0.2301025390625
    rbf 	 0.05 	 0.35009765625
    linear 	 1.0 	 0.199951171875
    rbf 	 1.0 	 0.159912109375
    linear 	 50 	 0.199951171875
    rbf 	 50 	 0.2099609375
    
    The best Classifier has kernel= rbf   and C= 1.0



![png](Cross%20Validation_files/Cross%20Validation_1_2.png)

