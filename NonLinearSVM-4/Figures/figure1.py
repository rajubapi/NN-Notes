from sklearn import svm
import sklearn.datasets as datasets
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np

def getClassifiers():
    return [svm.SVC(C=0.001,kernel='linear'),svm.SVC(C=0.001,kernel='rbf'),
        svm.SVC(C=0.05,kernel='linear'),svm.SVC(C=0.05,kernel='rbf'),

        svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),
        svm.SVC(C=50,kernel='linear'),svm.SVC(C=50,kernel='rbf')]

def getDataSets(x,y):
    list_A,list_B=[],[]
    for i in range(len(y)):
        if y[i]==0:
            list_A.append(x[i])
        else:
            list_B.append(x[i])
    return np.array(list_A),np.array(list_B)

def draw2D(dataset_A,dataset_B):
    plt.scatter(dataset_A[:,0],dataset_A[:,1],c="b",marker="o")
    plt.scatter(dataset_B[:,0],dataset_B[:,1],c="r",marker="o")
    plt.title("Dataset")
    plt.show()

def getAccuracy(predict,actual):
    correct=0
    for i in range(actual.shape[0]):
        if predict[i]==actual[i]:
            correct+=1
    return correct/actual.shape[0]

def performCrossValidation(classifiers,dataX,datay):
    numberOfFolds=5
    errorTrack=[]
    minError=9999
    for classifier in classifiers:
        kfold = KFold(numberOfFolds,False, 1)
        error=[]
        for train, test in kfold.split(dataX):
            trainingSet=dataX[train]
            testingSet=dataX[test]   
            lablesTrain=datay[train]
            lablesTest=datay[test]
            classifier.fit(trainingSet, lablesTrain)
            predicted=classifier.predict(testingSet)
            accuracy=getAccuracy(predicted,lablesTest)
            err=1-accuracy
            error.append(err)
        meanError=np.mean(np.array(error),dtype=np.float16)
        errorTrack.append([float(meanError),classifier.C,classifier.kernel])
        if minError > meanError:
                minError=meanError
                bestClassifier=classifier
    return np.array(errorTrack),bestClassifier


def plotGraph(input):
    x=input[:,1]
    y=input[:,0].astype(np.float32)
    plt.plot(x,y,"ro")
    plt.xlabel('C values')
    plt.ylabel('Mean Cross Validation Error')
    for i in range(len(input)):
        plt.annotate(input[i,2], (x[i],y[i]))
    plt.show()

def performCVandgetResult():
    X,y=datasets.make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.1, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)
    dataset_A,dataset_B=getDataSets(X,y)
    draw2D(dataset_A,dataset_B)
    classifiers=getClassifiers()
    errors,bestClassifier=performCrossValidation(classifiers,X,y)
    print("\nKernel\tC\tMean Cross Validation Error")
    print("---------------------------------------")
    for i in range(len(classifiers)):
        print(classifiers[i].kernel,"\t",classifiers[i].C,"\t",errors[i,0])
    print("\nThe best Classifier has kernel=",bestClassifier.kernel,"  and C=",bestClassifier.C)

    
    plotGraph(errors)