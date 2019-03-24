from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as datasets

def draw3D():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x,y=datasets.make_circles(n_samples=100, shuffle=True, noise=None, random_state=None, factor=0.5)
    dataset_A,dataset_B=getDataSets(x,y)
    ax.scatter(dataset_A[:,0],dataset_A[:,1],dataset_A[:,0]**2+dataset_A[:,1]**2,c="b",marker="o")
    ax.scatter(dataset_B[:,0],dataset_B[:,1],dataset_B[:,0]**2+dataset_B[:,1]**2,c="r",marker="o")

    #creating the plane
    point  = np.array([0.5, 0, 0.5])
    normal = np.array([0, 0, 0.5])
    d = -point.dot(normal)
    xx, yy = np.meshgrid(range(-1,2), range(-1,2))
    z = (-normal[0]*xx - normal[1]*yy - d) * 1. /normal[2]
    ax.plot_surface(xx, yy, z,alpha=0.5)

    plt.show()

def getDataSets(x,y):
    list_A,list_B=[],[]
    for i in range(len(y)):
        if y[i]==0:
            list_A.append(x[i])
        else:
            list_B.append(x[i])
    return np.array(list_A),np.array(list_B)

def draw2D():
    x,y=datasets.make_circles(n_samples=100, shuffle=True, noise=None, random_state=None, factor=0.5)
    dataset_A,dataset_B=getDataSets(x,y)
    plt.scatter(dataset_A[:,0],dataset_A[:,1],c="b",marker="o")
    plt.scatter(dataset_B[:,0],dataset_B[:,1],c="r",marker="o")
    plt.show()