import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def drawFeasibleRegion(data):
    x=np.arange(-10,10,1)
    y1=-1*(data[0][0]/data[0][1])*x
    y2=-1*(data[1][0]/data[1][1])*x
    y3=-1*(data[2][0]/data[2][1])*x
    y4=-1*(data[3][0]/data[3][1])*x
    y5=np.maximum(y1,y2)
    y6=np.minimum(y3,y4)
    y7= 10 + 0*x
    y8=np.maximum(y5,y6)
   
    #Plot the Data set
    plt.figure(num=None, figsize=(5, 4), dpi=100, facecolor='w', edgecolor='k')
    ax1=plt.scatter(x=data[:2,:1], y=data[:2,1:2], color='blue',
               marker='+', label='Data Set 1')
    ax2=plt.scatter(x=data[2:,:1], y=data[2:,1:2], color='red',
               marker='o', label='Data Set 2')
    for i in range(len(data)):
        txt="("+str(data[i][0])+","+str(data[i][1])+")"
        plt.annotate(txt,(data[i][0],data[i][1]))


    #Plot all the lines
    plt.plot(x,y1,linestyle="--")
    plt.plot(x,y2,linestyle="--")
    plt.plot(x,y3,linestyle="--")
    plt.plot(x,y4,linestyle="--")

    #Plot the vectors
    origin = [0], [0] # origin point
    for i in range(len(data)):
        datax=data[i][0]
        datay=data[i][1]
        plt.quiver(*origin, datax, datay, color=['b'], scale=21)


    #Shade the intersection region
    plt.fill_between(x,y8,y7,color="green",alpha='0.5')
    plt.xlim(-6,6)
    plt.ylim(-6,6)
    plt.grid(True)
    rect1 = Rectangle((0, 0), 1, 1, fc="g", alpha=0.5)
    plt.legend([rect1,ax1,ax2],["Feasible Region","Dataset 1","Dataset 2"])
    plt.show()

def draw():
    data = np.array([[-5,5],[5,5],[-5,-5],[5,-5]], dtype=np.float128)
    drawFeasibleRegion(data)    

draw()