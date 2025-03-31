import random
import mlp1 as mlp
from utils import *
import numpy as np
from collections import Counter

STUDENTS = [
    {"name": "YOUR NAME", "ID": "YOUR ID NUMBER"},
    {"name": "YOUR NAME", "ID": "YOUR ID NUMBER"},
]


def feats_to_vec(features):
    # YOUR CODE HERE.
    # Should return a numpy vector of features.

    bigram_count = Counter(features)

    x_vec = np.zeros(len(vocab))

    for i,bigram in enumerate(sorted(list(vocab))):
        x_vec[i] = bigram_count[bigram]

    return x_vec


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        
        label = L2I[label]

        x = feats_to_vec(features)

        pred = mlp.predict(x,params)

        if pred == label:
            good+=1
        else:
            bad+=1

    return good / (good + bad)


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in range(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features)  # convert features to a vector.

            y = L2I[label]  # convert the label to number if needed.

            loss, grads = mlp.loss_and_gradients(x, y, params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.
            W,b,U,b_tag = params

            gW, gb, gU, gb_tag = grads

            W = W - learning_rate * gW
            b = b - learning_rate * gb
            U = U - learning_rate * gU
            b_tag = b_tag - learning_rate * gb_tag

            params =  W,b,U,b_tag

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params


if __name__ == "__main__":
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.

    train_data = [(l,text_to_bigrams(t)) for l,t in read_data("data/train")]
    dev_data   = [(l,text_to_bigrams(t)) for l,t in read_data("data/dev")]

    fc = Counter()
    for l,feats in train_data:
        fc.update(feats)

    # 600 most common bigrams in the training set.
    vocab = set([x for x,c in fc.most_common(600)])

    # label strings to IDs
    L2I = {l:i for i,l in enumerate(list(sorted(set([l for l,t in train_data]))))}
    # feature strings (bigrams) to IDs
    F2I = {f:i for i,f in enumerate(list(sorted(vocab)))}

    in_dim = len(vocab)

    hidden_dim = 500

    out_dim = len(L2I)

    num_iterations = 20
    
    learning_rate = 0.001

    # ...

    params = mlp.create_classifier(in_dim,hidden_dim, out_dim)
    trained_params = train_classifier(
        train_data, dev_data, num_iterations, learning_rate, params
    )
