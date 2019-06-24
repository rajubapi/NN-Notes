import random
import math

class Network:
    '''
        Manage network
        size  = Board size
        unitsum = Entity The number of elements of the number of lines, the number in the element is the column position of the line
        unitout = 
        cost = cost
    '''
    def __init__(self, size):
        '''
        Initialize
        Initialize the network (units, weights, energy)
        '''
        #Unit initialization
        self.size  = size
        self.unit = [[1 if i == j else 0 for i in range(size)] for j in range(size)]   # list(range(size))

        # Weigtt setting
        p_a = 1
        p_b = 1
        p_c = 1
        p_d = 1

        self.weight = [[[[-2*p_a*self.nd(x,a)*self.d(y,b)-2*p_b*self.nd(y,b)*self.d(x,a)-2*p_c*self.d(x-y,a-b)-2*p_d*self.d(x+y,a+b) for x in range(size)] for y in range(size)] for a in range(size)] for b in range(size)]
        # Set self-coupling to 0
        for x in range(self.size):
            for y in range(self.size):
                self.weight[x][y][x][y] = 0

        self.theta = -2*(p_a+p_b)

    def d(self, i,j):
        '''
        delta function
        '''
        return 1 if i==j else 0

    def nd(self, i,j):
        '''
        Inversion of delta function
        '''
        return 0 if i==j else 1

    def update(self):
        '''
        Update the value of any unit
        hopfield continuous model
        '''
        rx = random.randrange(self.size)
        ry = random.randrange(self.size)

        val_in = -self.theta  # Input value to unit
        for a in range(self.size):
            for b in range(self.size):
                val_in += self.weight[rx][ry][a][b]*self.unit[a][b]
        val = (1.0+math.tanh(val_in)) / 2.0  # Sigmoid function

        if val != self.unit[rx][ry]:
            self.unit[rx][ry] = val
            return True

        return False

    def energy(self):
        '''
        Energy calculation  
        '''
        item1 = 0
        item2 = 0
        for x in range(self.size):
            for y in range(self.size):
                for a in range(self.size):
                    for b in range(self.size):
                        item1 += self.weight[x][y][a][b] * self.unit[x][y] * self.unit[a][b]
                item2 += self.theta * self.unit[x][y]
        return  -item1/2.0+item2 + self.size

    def is_active(self,val):
        '''
        Determine the existence of Queen from the specified value
        '''
        return val >= 0.5

    def check(self):
        '''
        Correct judgment
        '''
        checkunit = [[0 for i in range(self.size)] for j in range(self.size)]
        count = 0
        for x in range(self.size):
            for y in range(self.size):
                if self.is_active(self.unit[x][y]):
                    count += 1

                    # False if you meet
                    if checkunit[x][y] == 1:
                        return False

                    # Same y-axis
                    for cx in range(self.size):
                        checkunit[cx][y] = 1
                    # Same x-axis
                    for cy in range(self.size):
                        checkunit[x][cy] = 1
                    # Lower right
                    dl = y - x
                    for ix in range(self.size):
                        iy = ix + dl
                        if iy < 0:
                            continue
                        if iy >= self.size:
                            break
                        checkunit[ix][iy] = 1
                    # Top right
                    dl = x + y
                    for ix in range(self.size):
                        iy = dl - ix
                        if iy < 0:
                            break
                        if iy >= self.size:
                            continue
                        checkunit[ix][iy] = 1

        # True if no size queens meet
        return True if count == self.size else False

    def display(self):
        '''
        Express
        '''
        print(self.energy())
        for x in range(self.size):
            print( [ 1 if self.is_active(val) else 0 for val in self.unit[x] ] )
        print()


def train(size, max_iter):
    '''
    Learing
    '''
    network = Network(size)
    network.display()

    for iter in range(max_iter):
        if network.update():
            network.display()

            # Exit when solution is reached
            if network.check():
                print("OK")
                break
    return

if __name__ == '__main__':
    size = 8
#    max_iter = 170000
    max_iter = 1000
    train(size, max_iter)