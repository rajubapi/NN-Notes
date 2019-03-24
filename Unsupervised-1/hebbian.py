import numpy as np
import matplotlib.pyplot as plt
import imageio


def plot(weight):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(data1, data2)
    origin = [0], [0] # origin point
    ax.quiver(*origin, weight[:,0], weight[:,1],scale=8.0,scale_units="inches")
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return image
  


class NeuralNetwork:
    def __init__(self,data,weight,epochs=1):
        self.data=data
        self.weight=weight
        self.epochs=epochs

    def learn(self):
        images=[]
        for _ in range(self.epochs):
            for j in self.data:
                c=np.dot(self.weight,j.T)
                if(c>0):
                    c=0.01
                elif(c<0):
                    c=-0.01
                else:
                    c=0
                self.weight=self.weight+(c*j)
                images.append(plot(self.weight))
        imageio.mimwrite('./Gifs/hebbian1.gif', np.array(images), fps=1)

np.random.seed(1)
data1 = 20 * np.random.randn(100) 
data2 = data1 + (10 * np.random.randn(100))
data=np.stack((data1,data2),axis=1)
weight=np.array([[4,-2]])

nn=NeuralNetwork(data,weight)
nn.learn()