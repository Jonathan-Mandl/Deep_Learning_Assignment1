import numpy as np
from grad_check import gradient_check
from loglinear import softmax
STUDENTS = [
    {"name": "Danielle Hodaya Shrem", "ID": "208150433"},
    {"name": "Jonathan Mandl", "ID": "211399175"},
]


def classifier_output(x, params):
    """
    x: input vector
    params: list [W1, b1, W2, b2, ..., Wn, bn]
    """
    layer_input = x
    num_layers = len(params) // 2

    for i in range(num_layers - 1):
        W = params[2 * i]
        b = params[2 * i + 1]
        z = np.dot(layer_input, W) + b
        layer_input = np.tanh(z)  # activation

    # last layer (no tanh, only softmax)
    W_last = params[-2]
    b_last = params[-1]
    logits = np.dot(layer_input, W_last) + b_last
    exps = np.exp(logits - np.max(logits))
    probs = exps / np.sum(exps)
    return probs


def predict(x, params):
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """
    # Forward pass – store activations
    activations = [x]  # layer inputs (before W)
    pre_activations = []  # z = Wx + b before tanh

    num_layers = len(params) // 2
    layer_input = x

    for i in range(num_layers - 1):  # hidden layers
        W = params[2 * i]
        b = params[2 * i + 1]
        z = np.dot(layer_input, W) + b
        pre_activations.append(z)
        layer_input = np.tanh(z)
        activations.append(layer_input)

    # Final layer
    W_last = params[-2]
    b_last = params[-1]
    z_final = np.dot(layer_input, W_last) + b_last
    pre_activations.append(z_final)
    y_hat = softmax(z_final)
    activations.append(y_hat)

    # Loss
    y_true = np.zeros_like(y_hat)
    y_true[y] = 1
    epsilon = 1e-15
    loss = -np.sum(y_true * np.log(np.clip(y_hat, epsilon, 1.0)))

    # Backward pass
    grads = []
    delta = y_hat - y_true  # gradient of softmax+CE

    for i in reversed(range(num_layers)):
        a_prev = activations[i]
        W = params[2 * i]
        b = params[2 * i + 1]

        gW = np.outer(a_prev, delta)
        gb = delta

        grads.insert(0, gb)  # bias first
        grads.insert(0, gW)  # then W

        if i > 0:
            # Backprop through tanh
            dz = (1 - np.tanh(pre_activations[i - 1]) ** 2)
            delta = np.dot(delta, params[2 * i].T) * dz

    return loss, grads

def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.

    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    params = []
    for i in range(len(dims) - 1):
        W = np.random.randn(dims[i], dims[i + 1]) * 0.01
        b = np.zeros(dims[i + 1])
        params.append(W)
        params.append(b)
    return params


if __name__ == "__main__":
    dims = [3, 5, 4, 2]  # input -> 2 hidden layers -> output
    params = create_classifier(dims)
    x = np.array([1.0, 2.0, 3.0])
    y = 1  # label index

    for i in range(len(params)):
        def loss_and_param_grad(p):
            temp_params = list(params)  # shallow copy
            temp_params[i] = p
            loss, grads = loss_and_gradients(x, y, temp_params)
            return loss, grads[i]


        print(f"Checking gradient for param {i} ({'W' if i % 2 == 0 else 'b'})")
        gradient_check(loss_and_param_grad, params[i])