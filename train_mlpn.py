import numpy as np
import random
from utils import *
import mlpn
from collections import Counter




def feats_to_vec(features):
    bigram_count = Counter(features)
    x_vec = np.zeros(len(vocab))
    for i, bigram in enumerate(sorted(vocab)):
        x_vec[i] = bigram_count[bigram]
    return x_vec



def accuracy_on_dataset(dataset, params):
    good = 0
    for label, features in dataset:
        x = feats_to_vec(features)
        pred = mlpn.predict(x, params)
        if pred == L2I[label]:
            good += 1
    return good / len(dataset)


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    for i in range(num_iterations):
        cum_loss = 0.0
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features)
            y = L2I[label]
            loss, grads = mlpn.loss_and_gradients(x, y, params)

            for j in range(len(params)):
                params[j] -= learning_rate * grads[j]

            cum_loss += loss

        train_loss = cum_loss / len(train_data)
        train_acc = accuracy_on_dataset(train_data, params)
        dev_acc = accuracy_on_dataset(dev_data, params)
        print(f"{i}: loss={train_loss:.4f}, train_acc={train_acc:.4f}, dev_acc={dev_acc:.4f}")
    return params





if __name__ == "__main__":
    train_data = [(l, text_to_bigrams(t)) for l, t in read_data("data/train")]
    dev_data = [(l, text_to_bigrams(t)) for l, t in read_data("data/dev")]

    fc = Counter()
    for l, feats in train_data:
        fc.update(feats)

    vocab = set([x for x, c in fc.most_common(600)])
    L2I = {l: i for i, l in enumerate(sorted(set([l for l, _ in train_data])))}
    I2L = {i: l for l, i in L2I.items()}
    F2I = {f: i for i, f in enumerate(sorted(vocab))}

    in_dim = len(vocab)
    out_dim = len(L2I)

    dims = [in_dim, 128, 64, out_dim]
    learning_rate = 0.01
    num_iterations = 10

    params = mlpn.create_classifier(dims)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)
