import numpy as np
from grad_check import gradient_check

STUDENTS = [
    {"name": "Danielle Hodaya Shrem", "ID": "208150433"},
    {"name": "Jonathan Mandl", "ID": "211399175"},
]

def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    # YOUR CODE HERE
    # Your code should be fast, so use a vectorized implementation using numpy,
    # don't use any loops.
    # With a vectorized implementation, the code should be no more than 2 lines.
    #
    # For numeric stability, use the identify you proved in Ex 2 Q1.
    x = np.exp(x - np.max(x))
    return x / np.sum(x)


def classifier_output(x, params):
    # YOUR CODE HERE.
    W,b,U,b_tag = params

    z1 = tanh(np.add(np.dot(W.T,x),b))

    z2 = np.add(np.dot(U.T,z1),b_tag)

    probs = softmax(z2)

    return probs


def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))


def tanh(x):

    numerator = np.exp(x) - np.exp(-x)
    denominator =  np.exp(x) + np.exp(-x)

    return numerator/denominator


def tanh_derivative(x):
    return 1 - np.power(tanh(x), 2)


def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """
    # YOU CODE HERE
    W,b,U,b_tag = params

    n_out = U.shape[1]

    z1 = np.dot(W.T, x) + b
    h = tanh(z1)
    z2 = np.dot(U.T, h) + b_tag
    y_hat = softmax(z2)

    y_true = np.zeros(n_out)
    y_true[y] = 1

    epsilon = 1e-15

    loss = -np.dot(y_true, np.log(np.clip(y_hat, epsilon, 1.0)))

    gl_gz2 = y_hat - y_true

    gU = np.outer(h, gl_gz2) 

    gb_tag =  gl_gz2

    gz1 = np.dot(gl_gz2, U.T) * tanh_derivative(z1)

    x = np.array(x)

    gW = np.outer(x, gz1) 

    gb = gz1

    return loss, [gW, gb, gU, gb_tag]


def create_classifier(in_dim, hid_dim, out_dim):

    W = np.random.randn(in_dim, hid_dim) * 0.01 
    b = np.zeros(hid_dim)
    U = np.random.randn(hid_dim, out_dim) * 0.01
    b_tag = np.zeros(out_dim)
    return [W, b, U, b_tag]


if __name__ == "__main__":

    W, b, U, b_tag = create_classifier(3, 6, 2)

    def _loss_and_W_grad(W):
        global b, U, b_tag
        loss, grads = loss_and_gradients(np.array([1, 2, 3]), 0, [W, b, U, b_tag])
        return loss, grads[0]

    def _loss_and_b_grad(b):
        global W, U, b_tag
        loss, grads = loss_and_gradients(np.array([1, 2, 3]), 0, [W, b, U, b_tag])
        return loss, grads[1]

    def _loss_and_U_grad(U):
        global W, b, b_tag
        loss, grads = loss_and_gradients(np.array([1, 2, 3]), 0, [W, b, U, b_tag])
        return loss, grads[2]

    def _loss_and_b_tag_grad(b_tag):
        global W, b, U
        loss, grads = loss_and_gradients(np.array([1, 2, 3]), 0, [W, b, U, b_tag])
        return loss, grads[3]



    for _ in range(10):

        W = np.random.randn(W.shape[0], W.shape[1])
        b = np.random.randn(b.shape[0])
        U = np.random.randn(U.shape[0], U.shape[1])
        b_tag =np.random.randn(b_tag.shape[0])

        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)
        gradient_check(_loss_and_U_grad, U)
        gradient_check(_loss_and_b_tag_grad, b_tag)