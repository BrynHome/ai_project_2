from pandas import DataFrame, read_csv, concat, to_numeric


train_df = read_csv("data/training.csv")
train_df = train_df.sample(10000)
n = len(train_df["stars"])
train_df = train_df.rename(columns={"stars": "label"})
train_df = train_df.dropna()
train_df["label"] = train_df["label"].apply(lambda x : int(x) - 1)
train_df.to_csv("data/small_neural_train.csv", index_label=False)
