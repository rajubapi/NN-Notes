import numpy as np
import matplotlib.pyplot as plt
import imageio


class SOM:
    def __init__(self, x, y, alpha_start=0.6, seed=42):
        """ Initialize the SOM object with a given map size

        :param x: {int} width of the map
        :param y: {int} height of the map
        :param alpha_start: {float} initial alpha at training start
        :param seed: {int} random seed to use
        """
        np.random.seed(seed)
        self.x = x
        self.y = y
        self.shape = (x, y)
        self.sigma = x / 2.
        self.alpha_start = alpha_start
        self.alphas = None
        self.sigmas = None
        self.epoch = 0
        self.map = np.array([])
        self.indxmap = np.stack(np.unravel_index(
            np.arange(x * y, dtype=int).reshape(x, y), (x, y)), 2)
        self.distmap = np.zeros((self.x, self.y))
        self.winner_indices = np.array([])
        self.inizialized = False
        self.error = 0.  # reconstruction error

    def man_dist_pbc(self, m, vector, shape=(10, 10)):
        """ Manhattan distance calculation of coordinates with periodic boundary condition
        :param m: {numpy.ndarray} array / matrix
        :param vector: {numpy.ndarray} array / vector
        :param shape: {tuple} shape of the SOM
        :return: {numpy.ndarray} Manhattan distance for v to m
        """
        dims = np.array(shape)
        delta = np.abs(m - vector)
        delta = np.where(delta > 0.5 * dims, np.abs(delta - dims), delta)
        return np.sum(delta, axis=len(m.shape) - 1)

    def initialize(self, data):
        """ Initialize the SOM neurons

        :param data: {numpy.ndarray} data to use for initialization
        :param how: {str} how to initialize the map, available: 'pca' (via 4 first eigenvalues) or 'random' (via random
            values normally distributed like data)
        :return: initialized map in self.map
        """
        self.map = np.random.normal(np.mean(data), np.std(
            data), size=(self.x, self.y, len(data[0])))
        self.inizialized = True

    def winner(self, vector):
        """ Compute the winner neuron closest to the vector (Euclidean distance)

        :param vector: {numpy.ndarray} vector of current data point(s)
        :return: indices of winning neuron
        """
        indx = np.argmin(np.sum((self.map - vector) ** 2, axis=2))
        return np.array([indx // self.x, indx % self.y])

    def cycle(self, vector):
        """ Perform one iteration in adapting the SOM towards a chosen data point

        :param vector: {numpy.ndarray} current data point
        """
        w = self.winner(vector)
        # get Manhattan distance (with PBC) of every neuron in the map to the winner
        dists = self.man_dist_pbc(self.indxmap, w, self.shape)
        # smooth the distances with the current sigma
        h = np.exp(-(dists / self.sigmas[self.epoch])
                   ** 2).reshape(self.x, self.y, 1)
        # update neuron weights
        self.map -= h * self.alphas[self.epoch] * (self.map - vector)

        print("Epoch %i;    Neuron [%i, %i];    \tSigma: %.4f;    alpha: %.4f" %
              (self.epoch, w[0], w[1], self.sigmas[self.epoch], self.alphas[self.epoch]), end='\r')
        self.epoch = self.epoch + 1

    def fit(self, data, epochs=0):
        """ Train the SOM on the given data for several iterations

        :param data: {numpy.ndarray} data to train on
        :param epochs: {int} number of iterations to train; if 0, epochs=len(data) and every data point is used once
        :param decay: {str} type of decay for alpha and sigma. Choose from 'hill' (Hill function) and 'linear', with
            'hill' having the form ``y = 1 / (1 + (x / 0.5) **4)``
        :return: The list images of all the plots for each epoch
        """
        if not self.inizialized:
            self.initialize(data)
        if not epochs:
            epochs = len(data)
            indx = np.random.choice(
                np.arange(len(data)), epochs, replace=False)
        else:
            indx = np.random.choice(np.arange(len(data)), epochs)

        # get alpha and sigma decays for given number of epochs or for hill decay
        epoch_list = np.linspace(0, 1, epochs)
        self.alphas = self.alpha_start / (1 + (epoch_list / 0.5) ** 4)
        self.sigmas = self.sigma / (1 + (epoch_list / 0.5) ** 4)

        images = []

        for i in range(epochs):
            self.cycle(data[indx[i]])
            images.append(self.get_plot_image(data, i))
        print("\nWriting the gif file...")
        imageio.mimwrite('./Images/som-training.gif',
                         np.array(images), fps=15)
        print("Done")

    def get_plot_image(self, data, epoch):
        """ Get the image of a single plot to make the gif

        :param data; The data
        :param winner: The coordinates of the winning neuron
        returns the image of the plot
        """
        fig = plt.figure()
        # Plot the data
        plt.scatter(data[:, 0], data[:, 1], label='Data')
        # Plot the neurons
        plt.scatter(self.map[:, :, 0], self.map[:, :, 1], label='SOM Map')
        plt.legend()
        plt.title("Epoch: " + str(epoch))
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return image
