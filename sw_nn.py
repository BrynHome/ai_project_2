import torch
import pandas
import random


DEVICE = "cpu"


def accuracy_function(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc


class NeuralNet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        #TODO: change the in features to match the dimensions
        # after doing your specific feature extraction method.
        # i.e. you might have 100 features instead of 25.
        # You can also add other layers and experiment with architecture.
        self.linear_layer_stack = torch.nn.Sequential(
            torch.nn.Linear(in_features=25, out_features=10),
            torch.nn.Linear(in_features=10, out_features=5)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


if __name__ == "__main__":

    # Read 1% of the data for fast testing.
    # TODO change this to read the whole training set when you are ready.
    training = pandas.read_csv("data/training_experiment_1.csv", header=0, skiprows=lambda i: i > 0 and random.random() > 0.01)

    # Convert to tensor
    X = training.drop(["stars", "funny", "useful", "cool"], axis=1).values
    y = training[["stars"]].values
    y = y-1
    X = torch.from_numpy(X).type(torch.float)
    y = torch.from_numpy(y).type(torch.LongTensor)

    # 1. Create NN model.
    model = NeuralNet().to(DEVICE)

    # 2. Create loss function - cross entropy loss
    loss_fn = torch.nn.CrossEntropyLoss()

    # 3. Create optimizer - stochastic gradient descent
    # lr == learning rate == 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # 4. Train NN model.
    torch.manual_seed(42)
    # NOTE: change epochs to something like 20-50 most likely
    # when training on the whole training set.
    # It's probably going to be slow.
    epochs = 100
    X = X.to(DEVICE)
    y = torch.flatten(y.to(DEVICE))
    

    for epoch in range(epochs):
        model.train()

        y_logits = model(X)
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

        loss = loss_fn(y_logits, y)
        acc = accuracy_function(y_true=y, y_pred=y_pred)
        
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        model.eval()

        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}%")


