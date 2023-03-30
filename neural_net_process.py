from datasets import load_dataset
from pandas import read_csv

train_df = read_csv("data/training.csv")
n = len(train_df["stars"])
train_df = train_df.drop(["Unnamed: 0"], axis=1)
train_df = train_df.rename(columns={"stars": "label"})
for x in range(n):
    train_df["label"][x] = train_df["label"][x] - 1

train_df.to_csv("data/neural_train.csv", index_label=False)