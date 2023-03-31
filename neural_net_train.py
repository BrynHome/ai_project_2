import copy

import numpy as np
from sklearn import metrics
import pandas as pd
import tqdm
from joblib import load, dump
from bh_neural_net_simple import RegressModel, ClassifyModel
import torch
import random

from sklearn.model_selection import train_test_split

CLASSIFICATION_LABELS = ["stars"]
TARGET_LABELS = ["funny", "useful", "cool", "stars"]
MODEL_FILE_PREFIX_CLF = "./models/bh_clf_"
MODEL_FILE_PREFIX_RGR = "./models/bh_rgr_"
target = "stars"
DEVICE = "cpu"


def accuracy_function(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()  # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc


def met(pred):
    predict, labels = pred
    predict = np.argmax(predict, axis=1)
    precision = metrics.precision_score(labels, predict, average='weighted')
    recall = metrics.recall_score(labels, predict, average='weighted')
    f1 = metrics.f1_score(labels, predict, average='weighted')
    ac = metrics.accuracy_score(labels, predict)
    return {
        "Accuracy": ac,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


if __name__ == "__main__":

    # Read 1% of the data for fast testing.
    # TODO change this to read the whole training set when you are ready.
    training = pd.read_csv("data/output.csv", header=0, skiprows=lambda i: i > 0 and random.random() > 0.10)


    # Convert to tensor
    X_t = training.drop(["stars", "funny", "useful", "cool"], axis=1).values
    y_t = training[["stars"]].values
    y_t = y_t - 1
    x_train, x_test, y_train, y_test = train_test_split(X_t, y_t, train_size=0.7, shuffle=True)
    x_train = torch.from_numpy(x_train).type(torch.float)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)
    x_test = torch.from_numpy(x_test).type(torch.float)
    y_test = torch.from_numpy(y_test).type(torch.LongTensor)

    # 1. Create NN model.
    model = ClassifyModel(101, 5).to(DEVICE) # Experiment 1 Different hidden units

    # 2. Create loss function - cross entropy loss

    loss_fn = torch.nn.CrossEntropyLoss()

    # 3. Create optimizer - stochastic gradient descent
    # lr == learning rate == 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1) # Experiment 3 Change learning rate

    # 4. Train NN model.
    torch.manual_seed(42)
    # NOTE: change epochs to something like 20-50 most likely
    # when training on the whole training set.
    # It's probably going to be slow.
    epochs = 100
    batch = 10
    batch_s = torch.arange(0, len(x_train), batch)
    X = x_train.to(DEVICE)
    y = torch.flatten(y_train.to(DEVICE))
    x_test = x_test.to(DEVICE)
    y_test = torch.flatten(y_test.to(DEVICE))

    best_mse = np.inf
    best_weights = None
    hist = []

    for epoch in range(epochs):
        """
        model.train()
        with tqdm.tqdm(batch_s, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                x_b = X[start:start + batch]
                y_b = y[start:start + batch]
                y_l = model(x_b)
                loss = loss_fn(y_l, y_b)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                bar.set_postfix(mse=float(loss))
        model.eval()
        y_l = model(x_test)
        y_pred = torch.softmax(y_l, dim=1).argmax(dim=1)
        mse = loss_fn(y_l, y_test)
        mse = float(mse)
        hist.append(mse)
        acc = accuracy_function(y_true=y_test, y_pred=y_pred)
        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(model.linear_layer_stack.state_dict())
        """
        model.train()

        y_logits = model(X)
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

        loss = loss_fn(y_logits, y)
        acc = accuracy_function(y_true=y, y_pred=y_pred)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        model.eval()
        with torch.inference_mode():
            test_logit = model(x_test)
            test_pred = torch.softmax(test_logit, dim=1).argmax(dim=1)
            test_loss = loss_fn(test_logit, y_test)
            test_acc = accuracy_function(y_true=y_test, y_pred=test_pred)
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")
    #model.load_state_dict(best_weights)
    dump(model, f"{MODEL_FILE_PREFIX_CLF}{target}.joblib")
