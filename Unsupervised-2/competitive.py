import numpy as np
import matplotlib.pyplot as plt
import imageio
np.random.seed(1)
data1 = 20 * np.random.randn(1000) 
data2 = data1 + (10 * np.random.randn(1000))
data=np.stack((data1,data2),axis=1)
weight=np.array([[4,-2]])

def plot(weights):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(data1[:,0],data1[:,1])
    ax.scatter(data2[:,0],data2[:,1])
    ax.scatter(data3[:,0],data3[:,1])
    ax.scatter(weights[:,0],weights[:,1])
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return image
  


class NeuralNetwork:
    def __init__(self,data,weights,epochs=5):
        self.data=data
        self.weights=weights
        self.epochs=epochs

    def learn(self):
        images=[]
        for _ in range(self.epochs):
            for j in self.data:
                min=9999
                index=-1
                for i,val in enumerate(self.weights):
                    dist = np.linalg.norm(j-val)
                    if min>dist:
                        min=dist
                        index=i
                self.weights[index]=self.weights[index]+0.1*(j-self.weights[index])
                images.append(plot(self.weights))
        imageio.mimwrite('./Gifs/competitive1.gif', np.array(images), fps=1)
        

data1 = np.array([[5,5],[5,7],[6,6],[7,5],[7,6]])
data2 = np.array([[1,-4],[1,-5],[2,-6],[2,-4],[2,-5]])
data3 = np.array([[-5,4],[-5,7],[-6,6],[-7,4],[-7,6]])
weights=np.random.rand(3,2)


data=np.concatenate((data1,data2,data3))
nn=NeuralNetwork(data,weights)
nn.learn()