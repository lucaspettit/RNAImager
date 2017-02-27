import random
import numpy
from os.path import isfile, join
import sys

class perceptron(object):
    _w = []

    def __init__(self):
        p = join('res', 'w.dat')
        if isfile(p):
            f = open(p, 'r')
        else:
            raise (SystemError, 'w.dat not found')

        line = f.readline().strip('\n').split(' ')[:-1]
        self._w = [float(num) for num in line]
        f.close()

    def predict(self, x):
        guess = numpy.dot(x, self._w)
        if guess < 0:
            return -1
        elif guess > 0:
            return 1
        else:
            guess = random.randrange(0, 1)
            if guess > 0.5:
                return 1
            return -1


class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [numpy.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [numpy.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(numpy.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        evaluations = []
        if test_data is not None:
            n_test = len(test_data)
        else:
            n_test = 0
        n = len(training_data)
        best = 0
        brains = (None, None)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            eval = self.evaluate(test_data)
            evaluations.append(eval)
            if eval >= best:
                eta -= eta * 0.1
                best = eval
                brains = self.biases, self.weights
        return evaluations

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = numpy.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = numpy.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = numpy.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = numpy.dot(delta, activations[-l-1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(numpy.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return output_activations - y


def _trainPerceptron(td, w, i):
    for t in td:
        x = t[:-1]
        y = 1 if t[-1] == i else -1
        if y * numpy.dot(w, x) <= 0:
            w += (y * x)
    return w


def _perceptronError(w, i, td):
    num_errors = 0
    for t in td:
        label = 1 if t[-1] == i else -1
        feature = t[:-1]
        prediction = numpy.dot(feature, w)
        prediction /= abs(prediction)
        if prediction + label == 0:
            num_errors += 1
    return float(num_errors)/float(len(td))


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    v = numpy.exp(-z)
    return float(1)/(float(1) + v)


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def neighbors(state, rules):
    moves = []
    x, y, z = state
    rx, ry, rz = rules
    xmove, xmin, xmax = rx
    ymove, ymin, ymax = ry
    zmove, zmin, zmax = rz

    for i in (x - xmove, x, x + xmove):
        if i < xmin or i >= xmax:
            continue

        for j in (y - ymove, y, y + ymove):
            if j < ymin or j >= ymax:
                continue

            for k in (z - zmove, z, z + zmove):
                if k < zmin or k >= zmax:
                    continue
                if i == x and j == y and k == z:
                    continue
                moves.append([i, j, k])

    return moves


def splitVisited(world, nodes):
    visited = []
    unvisited = []
    for x, y, z in nodes:
        if x in world:
            if y in world[x]:
                if z in world[x][y]:
                    visited.append((world[x][y][z], [x, y, z]))
                else:
                    unvisited.append([x, y, z])
            else:
                unvisited.append([x, y, z])
        else:
            unvisited.append([x, y, z])
    return visited, unvisited


def getBestLocations(world):
    states = []
    mostAwesome = -99999
    for x in world.keys():
        for y in world[x].keys():
            for z in world[x][y].keys():
                num, score = world[x][y][z]
                if score > mostAwesome:
                    states = [[x, y, z]]
                    mostAwesome = score
                elif score == mostAwesome:
                    states.append([x, y, z])
    return states


def updateWorld(world, state, score):
    batch, eta, layer = state
    if batch in world:
        if eta in world[batch]:
            if layer in world[batch][eta]:
                num, value = world[batch][eta][layer]
                value = (score + (num * value)) / (num + 1)
                world[batch][eta][layer] = (num + 1, value)
            else:
                world[batch][eta][layer] = (1, score)
        else:
            world[batch][eta] = {}
            world[batch][eta][layer] = (1, score)
    else:
        world[batch] = {}
        world[batch][eta] = {}
        world[batch][eta][layer] = (1, score)

    num, value = world[batch][eta][layer]

    sys.stdout.write('\tstate: {0}\n'.format(str(state)))
    sys.stdout.write('\tnumber of visits: {0}\n'.format(str(num)))
    sys.stdout.write('\tstate score: {0}\n\n'.format(str(value)))

    return world


def randomState(rules):
    rx, ry, rz = rules
    xmove, xmin, xmax = rx
    ymove, ymin, ymax = ry
    zmove, zmin, zmax = rz
    return [random.randint(xmin, xmax), ymin + (random.random() * (ymax - ymin)), random.randint(zmin, zmax)]


def trainNetworkParams(train, test, iterations):
    """
    Uses a variation of Reinforcement Learning to
    identify which parameters will make the best
    neural network
    """

    world = {}  # store each state in the world
    rules = [[1, 5, 15], [0.3, 0.3, 8.0], [1, 2, 100]]

    for i in range(iterations):

        sys.stdout.write('Iteration {0}\n'.format(str(i)))

        states = getBestLocations(world)

        if len(states) < 2:
            while len(states) < 2:
                states.append(randomState(rules))
        else:
            states.append(randomState(rules))

        for state in states:
            for i in range(50):
                sys.stdout.write('step: {0} => {1}\n'.format(str(i), str(state)))

                n = Network([787, state[2], 2])
                res = numpy.array(n.SGD(train, 30, state[0], state[1], test))
                score = numpy.max(res)
                world = updateWorld(world, state, score)

                moves = neighbors(state, rules)
                visited, unvisited = splitVisited(world, moves)

                if len(unvisited) > 0:
                    state = unvisited[random.randint(0, len(unvisited) - 1)]
                    sys.stdout.write('unvisited\n')
                else:
                    best = []
                    for score, state in visited:
                        if len(best) == 0 or best[0][0] == score:
                            best.append((score, state))
                        else:
                            best = [(score, state)]

                    if len(best) == 0 or random.randint(0, 10) < 3:
                        state = visited[random.randint(0, len(visited) - 1)][1]
                        sys.stdout.write('visited\n')
                    else:
                        state = best[random.randint(0, len(best) - 1)][1]
                        sys.stdout.write('best\n')

    return getBestLocations(world)

if __name__ == "__main__":
    sys.stdout.write('training AI\n')
    file = open(join('res', join('training', 'data.dat')), 'r')
    train = [[float(x) for x in l.split(' ')] for l in file]
    file.close()
    feature_len = len(train[0]) - 1
    train = [[numpy.array(t[:-1]).reshape(feature_len, 1), int(t[-1])] for t in train]
    test = []
    train_length = int(len(train) * 0.90)
    while len(train) > train_length:
        index = random.randint(0, len(train)-1)
        test.append(train[index])
        del train[index]

    params = trainNetworkParams(train, test, 10)
    sys.stdout.write('best = {0}\n'.format(str(params)))
    sys.stdout.flush()

    #params = [10, 1.0, [787, 20, 2]]
    #net = Network(params[-1])
    #evals, brains = net.SGD(train, 30, params[0], params[1], test)
    #evals = numpy.array(evals)
    #variance = numpy.var(evals)
    #average = numpy.average(evals)
    #slope = evals[-1] - evals[0]
    #print('Average: ' + str(average))
    #print('Variance: ' + str(variance))
    #print('Slope: ' + str(slope))

    #fbias = open(join('res', join('training', 'biases.dat')), 'w')
    #fweight = open(join('res', join('training', 'weights.dat')), 'w')
    #file = open(join('res', join('training', 'network_params.dat')), 'w')

    #biases = ''
    #for bias in brains[0]:
    #    for b in bias:
    #        biases += str(b[0]) + ' '
    #    biases += '\n'
    #fbias.write(biases)
    #fbias.close()

    #weights = ''
    #for weight in brains[1]:
    #    for w in weight:
    #        for v in w:
    #            weights += str(v) + ' '
    #        weights += '\n'
    #fweight.write(weights)
    #fweight.close()

    #s = ''
    #for p in params:
    #    s += str(p) + ' '
    #file.write(s)
    #file.close()

