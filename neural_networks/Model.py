import numpy as np
from copy import deepcopy
import os

def one_hot_encoding(y_train, y_test):
    num_labels = len(y_train)
    train_labels = np.zeros((num_labels, 10))
    for i in range(num_labels):
        # y_train is a 1D array whose value is the target integer
        train_labels[i][y_train[i]] = 1
    
    num_labels = len(y_test)
    test_labels = np.zeros((num_labels, 10))
    for i in range(num_labels):
        # y_test is a 1D array whose value is the target integer
        test_labels[i][y_test[i]] = 1

    return train_labels, test_labels

import pickle

def load():
    os.chdir('Mnist_Data')
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    os.chdir('../')
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

def assert_same_shape(a, b):
    assert a.shape == b.shape

def eval_regression_model(net, x_test, y_test):
    preds = net.forward(x_test)

    mae = np.mean(np.abs(preds - y_test))
    rmse = np.sqrt(np.mean(np.power(preds - y_test, 2)))

    print(f"Mean Absolute Error: {mae:.3f}")
    print(f"Root Mean Squared Error: {rmse:.3f}")


class Operation():
    
    def __init__(self):
        pass

    def forward(self, input_):
        self.input_ = input_
        self.output = self._output()

        return self.output

    def backward(self, output_grad):
        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)

        return self.input_grad

    def _output(self):
        raise NotImplementedError()

    def _input_grad(self, output_grad):
        raise NotImplementedError()


class ParamOperation(Operation):
    
    def __init__(self, param):
        super().__init__()
        self.param = param

    def backward(self, output_grad):
        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)
        assert_same_shape(self.param, self.param_grad)

        return self.input_grad

    def _param_grad(self, output_grad):
        raise NotImplementedError()


class WeightsMultiply(ParamOperation):
    
    def __init__(self, W):
        super().__init__(W)

    def _output(self):
        return np.dot(self.input_, self.param)

    def _input_grad(self, output_grad):
        return np.dot(output_grad, np.transpose(self.param, (1, 0)))

    def _param_grad(self, output_grad):
        return np.dot(np.transpose(self.input_, (1, 0)), output_grad)


class BiasAdd(ParamOperation):

    def __init__(self, B):
        assert B.shape[0] == 1
        super().__init__(B)

    def _output(self):
        return self.input_ + self.param

    def _input_grad(self, output_grad):
        return np.ones_like(self.input_) * output_grad

    def _param_grad(self, output_grad):
        param_grad = np.ones_like(self.param) * output_grad
        return np.sum(param_grad, axis=0).reshape(1, -1)

class Sigmoid(Operation):
    
    def __init__(self):
        super().__init__()

    def _output(self):
        return 1.0 / (1.0 + np.exp(-1.0 * self.input_))

    def _input_grad(self, output_grad):
        sigmoid_grad = self.output * (1.0 - self.output)
        return sigmoid_grad * output_grad


class Tanh(Operation):

    def __init__(self):
        super().__init__()

    def _output(self):
        return np.tanh(self.input_)

    def _input_grad(self, output_grad):
        return output_grad * (1 - self.output * self.output)


class Linear(Operation):
    
    def __init__(self):
        super().__init__()

    def _output(self):
        return self.input_

    def _input_grad(self, output_grad):
        return output_grad

class Layer():
    
    def __init__(self, neurons):
        self.neurons = neurons
        self.first = True
        self.params = []
        self.param_grads = []
        self.operations = []

    def forward(self, input_):
        if self.first:
            self._setup_layer(input_)
            self.first = False

        self.input_ = input_

        for operation in self.operations:
            input_ = operation.forward(input_)
        self.output = input_

        return self.output

    def backward(self, output_grad):

        assert_same_shape(self.output, output_grad)

        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)

        input_grad = output_grad
        self._param_grads()
        return input_grad
        

    def _setup_layer(self, input_):
        raise NotImplementedError()

    def _params(self):
        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)

    def _param_grads(self):   
        self.param_grads = [] 
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

class Dense(Layer):
    
    def __init__(self, neurons, activation):
        super().__init__(neurons)
        self.activation = activation


    def _setup_layer(self, input_):
        self.params = []

        if self.seed:
            np.random.seed(self.seed)

        #Weights
        self.params.append(np.random.randn(input_.shape[1], self.neurons))

        #Bias
        self.params.append(np.random.randn(1, self.neurons))

        self.operations = [
            WeightsMultiply(self.params[0]),
            BiasAdd(self.params[1]),
            self.activation
            ]


class Loss():
    
    def __init__(self):
        pass

    def forward(self, predictions, y_test):
        '''
        Computes loss of the model
        '''
        assert_same_shape(predictions, y_test)

        self.preds = predictions
        self.target = y_test

        loss =  self._output()
        return loss
    
    def backward(self):
        '''
        computes gradient of loss with respect to its input
        '''
        return self._input_grad()

    def _output(self, preds):
        '''
        Instance subclass should already have their own defined _output() method
        '''
        raise NotImplementedError()

    def _input_grad(self):
        '''
        Instance subclass should already have their own defined _input_grad() method
        '''
        raise NotImplementedError()


class MeanSquaredError(Loss):
    
    def __init__(self):
        super().__init__()

    def _output(self):
        return np.sum(np.power(self.preds - self.target, 2)) / self.preds.shape[0]

    def _input_grad(self):
        return 2.0 * (self.preds - self.target) / self.preds.shape[0]


class NeuralNetwork():

    def __init__(self, layers, loss, seed):
        self.layers = layers
        self.loss = loss
        self.seed = seed
        if self.seed:
            for layer in self.layers:
                setattr(layer, "seed", self.seed)

    def forward(self, x_batch):
        '''
        Calculates predictions through a forward pass
        '''
        x_output = x_batch

        for layer in self.layers:
            x_output = layer.forward(x_output)

        return x_output

    def backward(self, output_grad):
        
        grad = output_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train_batch(self, x_batch, y_batch):
        '''
        Train batches by calling forward and backward passes
        '''
        
        preds = self.forward(x_batch)
        loss = self.loss.forward(preds, y_batch)

        loss_grad = self.loss.backward()
        self.backward(loss_grad)

        return loss

    def params(self):
        '''
        Yields paramter of each layer
        '''
        for layer in self.layers:
            yield from layer.params

    def param_grads(self):
        '''
        Yields parameter gradients of each layer
        '''
        for layer in self.layers:
            yield from layer.param_grads


class Optimizer():

    def __init__(self, init_learning_rate):
        self.init_learning_rate = init_learning_rate
        self.first = True

    def step(self):
        '''
        Instance subclass should already have their own defined .step() method
        '''
        raise NotImplementedError()


class SGD(Optimizer):

    def __init__(self, init_learning_rate):
        super().__init__(init_learning_rate)

    def step(self):
        '''
        Updates weights and biases of the neural network
        '''

        for param, param_grad in zip(self.net.params(), self.net.param_grads()):
            param -= self.init_learning_rate * param_grad


class Trainer():
    
    def __init__(self, net, optim):
        self.net = net
        self.optim = optim
        setattr(self.optim, "net", self.net)

    def permute_data(self, x_train, y_train):
        '''
        shuffles data into a random order
        '''
        assert x_train.shape[0] == y_train.shape[0]

        perm = np.random.permutation(x_train.shape[0])
        return x_train[perm], y_train[perm]

    def generate_batches(self, x_train, y_train, batch_size):
        '''
        yields batches of data of size batch_size
        '''

        for i in range(0, x_train.shape[0], batch_size):
            x_batch = x_train[i: i + batch_size]
            y_batch = y_train[i: i + batch_size]

            yield x_batch, y_batch

    def fit(self, x_train, y_train, x_test, y_test,
            epochs,
            eval_every,
            batch_size,
            seed,
            restart=True):

        np.random.seed(seed)
        
        greatest_loss  = 1e9

        if restart:
            for layer in self.net.layers:
                layer.first = True

        for epoch in range(epochs):

            if (epoch+1) % eval_every == 0:
                previous_model = deepcopy(self.net)

            # Shuffle data and make batches
            x_train, y_train = self.permute_data(x_train, y_train)
            batches = self.generate_batches(x_train, y_train, batch_size)

            for x_batch, y_batch in batches:

                self.net.train_batch(x_batch, y_batch)
                self.optim.step()

            if (epoch+1) % eval_every == 0:

                preds = self.net.forward(x_test)
                loss = self.net.loss.forward(preds, y_test)


                if loss < greatest_loss:
                    greatest_loss = loss
                    print(f"Validation loss at epoch {epoch+1} is {loss:.3f}.")
                else:
                    print(f"Loss increased after epoch {epoch+1}, final loss was {greatest_loss:.3f}; using model from epoch {epoch+1-eval_every}")
                    self.net = previous_model
                    setattr(self.optim, "net", self.net)

# ------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # from sklearn.datasets import load_boston

    # boston = load_boston()

    # data = boston.data
    # target = boston.target

    # from sklearn.preprocessing import StandardScaler

    # s = StandardScaler()
    # data = s.fit_transform(data)

    # from sklearn.model_selection import train_test_split

    # x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=.3, random_state=80718)
    # y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
    
    x_train, y_train, x_test, y_test = load()
    y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
    # train_labels, test_labels = one_hot_encoding(y_train, y_test)

    # linear_regression = NeuralNetwork(
    #     layers=[Dense(1, activation=Linear())],
    #     loss=MeanSquaredError(),
    #     seed=20190501)

    # neural_network = NeuralNetwork(
    #     layers=[
    #         Dense(neurons=13, activation=Sigmoid()),
    #         Dense(neurons=1, activation=Linear())],
    #     loss=MeanSquaredError(),
    #     seed=20190501)


    deep_neural_network = NeuralNetwork(
        layers=[
            Dense(neurons=784, activation=Sigmoid()),
            Dense(neurons=13, activation=Sigmoid()),
            Dense(neurons=1, activation=Linear())],
        loss=MeanSquaredError(),
        seed=20190501)

    deep_trainer = Trainer(
        deep_neural_network,
        SGD(init_learning_rate=0.01))

    print()
    print("Deep Neural Regression")
    print()
    deep_trainer.fit(
        x_train, y_train, x_test, y_test,
        epochs = 50,
        eval_every = 10,
        batch_size = 32,
        seed = 20190501,
        restart=True)
    print()
    eval_regression_model(deep_neural_network, x_test, y_test)