import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def drawVectorAndHyperPlane(data,desired):
    origin = [0], [0] # origin point
    
    for i in range(len(data)):
        datax=data[i][0]
        datay=data[i][1]
        slope=-1*(datax/datay)
        x=np.arange(-10,10,0.001)
        y=slope*x
        subplt=plt.subplot(2,2,i+1)
        labl="("+str(datax)+","+str(datay)+")"+"Vector x"+str(i+1)
        subplt.set_title(labl)
        plt.quiver(*origin, datax, datay, color=['r','b','g'], scale=21)
        plt.plot(x,y)
        rect1 = Rectangle((0, 0), 1, 1, fc="g", alpha=0.5)
        rect2 = Rectangle((0, 0), 1, 1, fc="r", alpha=0.5)  
        if(desired[i]==1):
            plt.fill_between(x,y,10, where=y>=slope*x, facecolor='green',alpha=0.5)
            plt.fill_between(x,y,-10, where=y<=slope*x, facecolor='red',alpha=0.5)
            plt.legend([rect1,rect2],["Right side","Wrong Side"],loc="lower right")

        else:
            plt.fill_between(x,y,10, where=y<=slope*x, facecolor='green',alpha=0.5)
            plt.fill_between(x,y,-10, where=y>=slope*x, facecolor='red',alpha=0.5)
        
            plt.legend([rect1,rect2],["Right side","Wrong Side"],loc="upper right")
    plt.show()

def draw():
    data = np.array([[-5,5],[5,5],[-5,-5],[5,-5]], dtype=np.float128)
    desired=np.array([1,1,0,0], dtype=np.float128)
    drawVectorAndHyperPlane(data,desired)

