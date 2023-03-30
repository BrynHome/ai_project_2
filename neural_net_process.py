from pandas import read_csv

train_df = read_csv("data/training.csv")
n = len(train_df["stars"])
train_df = train_df.rename(columns={"stars": "label"})
train_df = train_df.dropna()
train_df["label"] = train_df["label"].apply(lambda x : int(x) - 1)

train_df.to_csv("data/neural_train.csv", index_label=False)