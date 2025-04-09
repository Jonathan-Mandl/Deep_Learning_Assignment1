import mlp1 as mlp
import numpy as np
from xor_data import data

L2I = {0: 0, 1: 1}  # 0 או 1
I2L = {0: 0, 1: 1}

train_data = [(L2I[y], x) for y, x in data]

def feats_to_vec(x):
    return np.array(x)

def accuracy_on_dataset(dataset, params):
    correct = 0
    for label, x in dataset:
        pred = mlp.predict(feats_to_vec(x), params)
        if pred == label:
            correct += 1
    return correct / len(dataset)

def train_classifier(train_data, num_iterations, learning_rate, params):
    for i in range(num_iterations):
        if i == 0:
            print("\nPredictions on XOR examples at iter 0:")
            for label, x in train_data:
                x_vec = feats_to_vec(x)
                pred = mlp.predict(x_vec, params)
                probs = mlp.classifier_output(x_vec, params)
                print(f"Input: {x}, Label: {label}, Prediction: {pred}, Probs: {probs}")

        total_loss = 0
        for label, x in train_data:
            loss, grads = mlp.loss_and_gradients(feats_to_vec(x), label, params)
            W, b, U, b_tag = params
            gW, gb, gU, gb_tag = grads
            W -= learning_rate * gW
            b -= learning_rate * gb
            U -= learning_rate * gU
            b_tag -= learning_rate * gb_tag
            params = [W, b, U, b_tag]
            print(f"Loss: {loss:.4f}, ||gW||: {np.linalg.norm(gW):.4f}, ||gU||: {np.linalg.norm(gU):.4f}")

        acc = accuracy_on_dataset(train_data, params)
        print(f"Iter {i} | Accuracy: {acc}")
        if acc == 1.0:
            print(f"Solved XOR in {i+1} iterations")
            break
    return i + 1

if __name__ == "__main__":
    in_dim = 2
    hid_dim = 10
    out_dim = 2
    params = mlp.create_classifier(in_dim, hid_dim, out_dim)
    print("Initial weights:")
    W, b, U, b_tag = params
    print("W:\n", W)
    print("b:\n", b)
    print("U:\n", U)
    print("b_tag:\n", b_tag)

    train_classifier(train_data, 1000, 1, params)
