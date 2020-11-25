import numpy as np
from copy import deepcopy
from scipy.special import logsumexp
import pickle


def calc_model_accuracy(model, x_test, y_test):
    y_test = y_test.reshape(y_test.shape[0],)
    amount_right = np.equal(np.argmax(model.forward(x_test, inference=True), axis=1), y_test).sum()
    print(f'The model validation accuracy is: {amount_right * 100/ x_test.shape[0]:.2f}%')


def load():
    path = "/home/joshua/Desktop/python/DeepLearningFromScratch/MnistData/mnist.pkl"
    # path = './MnistData/mnist.pkl'
    with open(path,'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


def assert_same_shape(a, b):
    assert a.shape == b.shape

def to_2d_np(a, type='col'):

    assert a.ndim == 1
    
    if type == "col":        
        return a.reshape(-1, 1)
    elif type == "row":
        return a.reshape(1, -1)

class Operation():

    def __init__(self):
        pass

    def forward(self, input_, inference):
        self.input_ = input_
        self.output = self._output(inference)

        return self.output

    def backwards(self, output_grad):
        assert_same_shape(self.output, output_grad)
        self.input_grad = self._input_grad(output_grad)
        assert_same_shape(self.input_, self.input_grad)

        return self.input_grad

    def _output(self, inference):
        raise NotImplementedError

    def _input_grad(self, output_grad):
        raise NotImplementedError


class ParamOperation(Operation):

    def __init__(self, param):
        super().__init__()
        self.param = param

    def backwards(self, output_grad):
        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)
        assert_same_shape(self.param, self.param_grad)

        return self.input_grad

    def param_grad(self, output_grad):
        raise NotImplementedError


class WeightsMultiply(ParamOperation):

    def __init__(self, weights):
        super().__init__(weights)

    def _output(self, inference):
        return np.dot(self.input_, self.param)

    def _input_grad(self, output_grad):
        return np.dot(output_grad, np.transpose(self.param, (1, 0)))

    def _param_grad(self, output_grad):
        return np.dot(np.transpose(self.input_, (1, 0)), output_grad)


class BiasAdd(ParamOperation):

    def __init__(self, bias):
        assert bias.shape[0] == 1
        super().__init__(bias)

    def _output(self, inference):
        return self.input_ + self.param

    def _input_grad(self, output_grad):
        return np.ones_like(self.input_) * output_grad

    def _param_grad(self, output_grad):
        param_grad = np.ones_like(self.param) * output_grad
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])


class Tanh(Operation):

    def __init__(self):
        super().__init__()

    def _output(self, inference):
        return np.tanh(self.input_)

    def _input_grad(self, output_grad):
        return output_grad * (1 - self.output * self.output)


class Sigmoid(Operation):

    def __init__(self):
        super().__init__()

    def _output(self, inference):
        return 1.0/(1.0 + np.exp(-1.0 * self.input_))

    def _input_grad(self, output_grad):
        return (self.output * (1.0 - self.output)) * output_grad


class Linear(Operation):

    def __init__(self):
        super().__init__()

    def _output(self, inference):
        return self.input_

    def _input_grad(self, output_grad):
        return output_grad


class Layer():

    def __init__(self, neurons):
        self.neurons = neurons
        self.first = True
        self.operations = None
        self.params = None
        self.param_grads = []


    def setupLayer(self):
        raise NotImplementedError

    def forward(self, input_, inference=False):

        if self.first:
            self.setupLayer(input_)
            self.first = False

        for operation in self.operations:
            input_ = operation.forward(input_, inference)
        self.output = input_

        return self.output

    def backwards(self, output_grad):
        assert_same_shape(self.output, output_grad)

        for operation in reversed(self.operations):

            output_grad = operation.backwards(output_grad)

        self._param_grads()

        return output_grad

    def _param_grads(self):
        '''
        puts layer's param_grads into class variable
        '''
        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)


class Dense(Layer):

    def __init__(self, neurons, activation, dropout=1.0, weight_init=None):
        super().__init__(neurons)
        self.weight_init = weight_init
        self.dropout = dropout
        self.activation = activation

    def setupLayer(self, input_):
        if self.weight_init == 'glorot':
            scale = 2/(input_.shape[0] + self.neurons)
        else:
            scale = 1.0

        self.params = []

        self.params.append(np.random.normal(scale=scale, size=(input_.shape[1], self.neurons))) # weights
        self.params.append(np.random.normal(scale=scale, size=(1, self.neurons))) # bias

        self.operations = [
            WeightsMultiply(self.params[0]),
            BiasAdd(self.params[1]),
            self.activation]
        
        if self.dropout < 1.0:
            self.operations.append(Dropout(self.dropout))


class Loss():

    def __init__(self):
        pass

    def forward(self, prediction, target):
        assert_same_shape(prediction, target)

        self.prediction = prediction
        self.target = target

        return self._output()

    def backwards(self):
        input_grad = self._input_grad()
        assert_same_shape(self.prediction, input_grad)

        return input_grad

    def _output(self):
        raise NotImplementedError

    def _input_grad(self):
        raise NotImplementedError


class MeanSquaredError(Loss):
    
    def __init__(self):
        super().__init__()

    def _output(self):
        return np.sum(np.power(self.prediction - self.target, 2)) / self.prediction.shape[0]

    def _input_grad(self):
        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]


class SoftmaxCrossEntrophy(Loss):

    def __init__(self, min=1e-9):
        super().__init__()
        self.min = min
        self.single_output = False

    def softmax(self, x, axis=None):
        return np.exp(x - logsumexp(x, axis=axis, keepdims=True))

    def _output(self):

        #apply softmax function to each row in observation
        softmax_preds = self.softmax(self.prediction, axis=1)

        #clip maximum and minimum values
        self.softmax_preds = np.clip(softmax_preds, self.min, 1-self.min)

        smce_loss = (
            -1.0 * self.target * np.log(self.softmax_preds) - \
            (1.0 - self.target) * np.log(1.0 - self.softmax_preds)
        )
        return np.sum(smce_loss) / self.prediction.shape[0]

    
    def _input_grad(self):
        # returns average loss for each image (row)
        return (self.softmax_preds - self.target) / self.prediction.shape[0]


class Dropout(Operation):

    def __init__(self, keep_prob):
        super().__init__()
        self.keep_prob = keep_prob

    def _output(self, inference):
        if inference:
            return self.input_ * self.keep_prob
        else:
            self.mask = np.random.binomial(1, self.keep_prob, size=self.input_.shape)

            return self.input_ * self.mask
    
    def _input_grad(self, output_grad):
        return output_grad * self.mask


class NeuralNetwork():

    def __init__(self, layers, loss, seed=None):
        self.layers = layers
        self.loss = loss
        self.seed = seed

        if self.seed:
            for layer in self.layers:
                setattr(layer, 'seed', self.seed)

    def forward(self, input_, inference):
        
        for layer in self.layers:
            input_ = layer.forward(input_, inference)

        return input_

    def backwards(self, output_grad):

        for layer in reversed(self.layers):
            output_grad = layer.backwards(output_grad)

        return output_grad

    def train_batch(self, x_batch, y_batch, inference=False):

        prediction = self.forward(x_batch, inference)
        loss_value = self.loss.forward(prediction, y_batch)
        self.backwards(self.loss.backwards())

        return loss_value

    def params(self):
        for layer in self.layers:
            yield from layer.params

    def param_grads(self):
        for layer in self.layers:
            yield from layer.param_grads


class Optimizer():

    def __init__(self, lr, final_lr, decay_type='exponential'):
        self.lr = lr
        self.final_lr = final_lr
        self.decay_type = decay_type
        self.first = True

    def step(self):
        raise NotImplementedError

    def _setup_decay(self, max_epochs):
        
        if not self.decay_type:
            return

        elif self.decay_type == 'exponential':
            # self.max_epochs defined in Trainer class
            self.decay_per_epoch = np.power(self.final_lr/self.lr, 1.0/(max_epochs-1))
        elif self.decay_type == 'linear':
            self.decay_per_epoch = (self.lr - self.final_lr) / (max_epochs-1)
    
    def _decay_lr(self):
        
        if not self.decay_type:
            return
        
        elif self.decay_type == 'exponential':
            self.lr *= self.decay_per_epoch
        elif self.decay_type == 'linear':
            self.lr -= self.decay_per_epoch


class SGD(Optimizer):

    def __init__(self, lr):
        super().__init__(lr)

    def step(self):

        for (param, param_grad) in zip(self.net.params(), self.net.param_grads()):
            param -= self.lr * param_grad


class SGDMomentum(Optimizer):

    def __init__(self, lr, final_lr, momentum, decay_type):
        super().__init__(lr, final_lr, decay_type)
        self.momentum = momentum


    def step(self):

        if self.first:
            self.velocities = [np.zeros_like(param) for param in self.net.params()]
            self.first = False

        for (param, param_grad, velocity) in zip(self.net.params(), self.net.param_grads(), self.velocities):
            self._update_rule(param=param, grad=param_grad, velocity=velocity)

    def _update_rule(self, param, grad, velocity):
        velocity *= self.momentum
        velocity += grad * self.lr
        param -= velocity


class Trainer():

    def __init__(self, net, optim):
        self.net = net
        self.optim = optim
        setattr(optim, 'net', self.net)

    def permute_data(self, x_train, y_train):
        perm = np.random.permutation(x_train.shape[0])
        return x_train[perm], y_train[perm]

    def generate_batches(self, x_train, y_train, batch_size):
        assert x_train.shape[0] == y_train.shape[0]

        for i in range(0, x_train.shape[0], batch_size):
            x_batch = x_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            yield x_batch, y_batch

    def fit(self, x_train, y_train, x_test, y_test, batch_size, epochs, eval_every, seed=None):
        
        if seed:
            for layer in self.net.layers:
                layer.seed = seed

        self.optim._setup_decay(epochs)
        greatest_loss = 1e9

        for epoch in range(epochs):

            if (epoch+1) % eval_every == 0:
                last_model = deepcopy(self.net)

            x_train, y_train = self.permute_data(x_train, y_train)
            batch_generator = self.generate_batches(x_train, y_train, batch_size)

            for x_batch, y_batch in batch_generator:

                self.net.train_batch(x_batch, y_batch)
                self.optim.step()

            self.optim._decay_lr()
            
            if (epoch+1) % eval_every == 0:
                preds = self.net.forward(x_test, inference=True)
                loss = self.net.loss.forward(preds, y_test)

                if loss < greatest_loss:
                    greatest_loss = loss
                    print(f'Validation loss for epoch {epoch+1}: {loss:.3f}')
                else:
                    print(f'Validation loss increased after epoch {epoch+1}, previous loss was {greatest_loss:.3f}; using model from epoch {epoch+1-eval_every}')
                    self.net = last_model
                    setattr(self.optim, 'net', self.net)


mnist_deepNet = NeuralNetwork(
    layers=[
        Dense(neurons=178, activation=Tanh(), weight_init='glorot', dropout=.8),
        Dense(neurons=46, activation=Tanh(), weight_init='glorot', dropout=.8),
        Dense(neurons=10, activation=Linear(), weight_init='glorot')],
    loss=SoftmaxCrossEntrophy(),
    seed=20190119)
    
optim = SGDMomentum(lr=0.2, final_lr=.05, momentum=0.9, decay_type='exponential')

trainer = Trainer(mnist_deepNet, optim)


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load()
    y_train, y_test = to_2d_np(y_train), to_2d_np(y_test)

    num_labels = len(y_train)
    train_labels = np.zeros((num_labels, 10))
    for i in range(num_labels):
        train_labels[i][y_train[i]] = 1

    num_labels = len(y_test)
    test_labels = np.zeros((num_labels, 10))
    for i in range(num_labels):
        test_labels[i][y_test[i]] = 1


    def scaleData(x_train, x_test):
        trainMean = np.mean(x_train)
        trainStd = np.std(x_train)
        return (x_train - trainMean) / trainStd, (x_test - trainMean) / trainStd

    # scaleData
    x_train, x_test = scaleData(x_train, x_test)

    mnist_deepNet = NeuralNetwork(
    layers=[
        Dense(neurons=178, activation=Tanh(), weight_init='glorot', dropout=.8),
        Dense(neurons=46, activation=Tanh(), weight_init='glorot', dropout=.8),
        Dense(neurons=10, activation=Linear(), weight_init='glorot')],
    loss=SoftmaxCrossEntrophy(),
    seed=20190119)
    
    optim = SGDMomentum(lr=0.2, final_lr=.05, momentum=0.9, decay_type='exponential')

    trainer = Trainer(mnist_deepNet, optim)
    trainer.fit(x_train, train_labels, x_test, test_labels,
            epochs = 50,
            eval_every = 10,
            batch_size = 60)
    print()
    calc_model_accuracy(mnist_deepNet, x_test, y_test)