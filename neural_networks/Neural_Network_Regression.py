'''
Mainly from Deep Learning from Scratch
Uses the Boston data set from scikit-learn which has thirteen different features
input size of 13 neurons, hidden layer of 13 neurons, and one output neuron
'''

#imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

boston = load_boston()
data = boston.data
target = boston.target
S = StandardScaler()
data = S.fit_transform(data)

def sigmoid(x):
    '''
    Sigmoid function
    '''
    return 1 / (1+np.exp(-x))


def init__weights(n_size, hidden_layer_size):
    '''
    Initializes weights with random values
    '''

    weights = {}
    weights['W1'] = np.random.randn(n_size, hidden_layer_size)
    weights['B1'] = np.random.randn(1, hidden_layer_size)
    weights['W2'] = np.random.randn(n_size, 1)
    weights['B2'] = np.random.randn(1, 1)

    return weights


def permute_data(X, y):
    '''
    permutates data for random order
    '''
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]


def generate_batch(X, y, start, batch_size):
    '''
    Cutes data into batches of given size
    '''
    
    # assert that X and y are two dimensional matricies
    assert X.ndim == y.ndim == 2

    if start + batch_size > X.shape[0]:
        batch_size = X.shape[0] - start

    X_batch = X[start: start+batch_size]
    y_batch = y[start: start+batch_size]

    return X_batch, y_batch


def forward_pass(X, y, weights):
    '''
    Computes information in a forward pass of the model
    '''

    N1 = np.dot(X, weights['W1'])
    P1 = N1 + weights['B1']
    O1 = sigmoid(P1)
    N2 = np.dot(O1, weights['W2'])
    P2 = N2 + weights['B2']
    
    loss = np.mean(np.power(y - P2, 2))

    forward_info = {}
    forward_info['X'] = X
    forward_info['N1'] = N1
    forward_info['P1'] = P1
    forward_info['O1'] = O1
    forward_info['N2'] = N2
    forward_info['P2'] = P2
    forward_info['y'] = y

    return forward_info, loss

def gradients(forward_info, weights):
    '''
    Computes gradients of the model
    dL_dW1 = dL_dP2 * dP2_dN2 * dN2_dO1 * dO1_dP1 * dP1_dN1 * dN1_dW1
    '''

    # b_size = batch_size
    # n = number of neurons in hidden layer

    dL_dP2 = -2 * (forward_info['y'] - forward_info['P2']) #shape = (b_size, 1)

    # -------------------------------Partials of dP2 -------------------------------

    dP2_dB2 = np.ones_like(weights['B2']) #shape (1, 1)
    dP2_dN2 = np.ones_like(forward_info['N2']) #shape (b_size, 1)
    
    # Connect back to dL

    dL_dB2 = (dL_dP2 * dP2_dB2).sum(axis=0) # shape (b_size, 1) -> (1,)
    dL_dN2 = dL_dP2 * dP2_dN2 #shape (b_size, 1)

    # -------------------------------Partials of dN2 -------------------------------

    dN2_dW2 = np.transpose(forward_info['O1'], (1, 0)) # shape (n, b_size)
    dN2_dO1 = np.transpose(weights['W2'], (1, 0)) # shape (1, n)

    # Connect back to dL

    dL_dW2 = np.dot(dN2_dW2, dL_dN2) # shape (n, 1)
    dL_dO1 = np.dot(dL_dN2, dN2_dO1) # shape (b_size, n)

    # -------------------------------Derivative of dO1 -------------------------------
    # derivative of sigmoid o: o' = o * (1-o)

    dO1_dP1 = sigmoid(forward_info['P1']) * (1 - sigmoid(forward_info['P1'])) # shape (b_size, n)

    # Connect back to dL

    dL_dP1 = dL_dO1 * dO1_dP1 # shape (b_size, n)

    # -------------------------------Partials of dP1 -------------------------------

    dP1_dB1 = np.ones_like(weights['B1']) # shape (n, n)
    dP1_dN1 = np.ones_like(forward_info['N1']) #shape (b_size, n)

    # Connect back to dL

    dL_dB1 = (dL_dP1 * dP1_dB1).sum(axis=0) # shape (b_size, n) -> (n,)
    dL_dN1 = dL_dP1 * dP1_dN1 #shape(b_size, n)

    # -------------------------------Parital of dN1 -------------------------------

    dN1_dW1 = np.transpose(forward_info['X'], (1, 0)) # shape (n, b_size)

    # Connect back to dL

    dL_dW1 = np.dot(dN1_dW1, dL_dN1) # shape (n, n)

    # -----------------------------------------------------------------------------

    # return dictionary of gradient losses
    gradient_loss = {}
    gradient_loss['W1'] = dL_dW1
    gradient_loss['B1'] = dL_dB1.sum(axis=0)
    gradient_loss['W2'] = dL_dW2
    gradient_loss['B2'] = dL_dB2.sum(axis=0)

    return gradient_loss


def train(X, y, iterations, learning_rate, hidden_layer_size, batch_size):
    '''
    Trains input data
    '''

    # Initialize weights
    weights = init__weights(X.shape[1], hidden_layer_size)

    # Permute data
    X, y = permute_data(X, y)
    
    losses = []
    start = 0

    for i in range(iterations):

        if start >= X.shape[0]:
            X, y = permute_data(X, y)
            start = 0


        X_batch, y_batch = generate_batch(X, y, start, batch_size)
        start += batch_size

        # pass forward the information
        forward_info, loss = forward_pass(X_batch, y_batch, weights)

        losses.append(loss)

        # Get the gradients
        gradient_loss = gradients(forward_info, weights)

        # Update the weights and biases
        for key in weights.keys():
            weights[key] -= learning_rate * gradient_loss[key]

    return losses, weights


def predict(X, weights):
    '''
    Returns an array of prediction values P2 of X_test values
    '''
    N1 = np.dot(X, weights['W1'])
    P1 = N1 + weights['B1']
    O1 = sigmoid(P1)
    N2 = np.dot(O1, weights['W2'])
    P2 = N2 + weights['B2']

    return P2

def mean_abolute_error(preds, y_test):
    return np.mean(np.abs(y_test - preds))

def root_mean_squared_error(preds, y_test):
    return np.sqrt(np.mean(np.power(y_test - preds, 2)))



if __name__ == '__main__':
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=.3)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    losses, weights = train(X_train, y_train, iterations=10000, learning_rate=.001, hidden_layer_size=13, batch_size=23)

    preds = predict(X_test, weights)

    mbe = mean_abolute_error(preds, y_test)
    rmse = root_mean_squared_error(preds, y_test)

    print(f"Mean Absolute Error:\n{mbe}")
    print(f"Root Mean Squared Error:\n{rmse}")