from pandas import DataFrame, read_csv, concat, to_numeric
REGRESSION_LABELS = ["funny", "useful", "cool"]

train_df = read_csv("data/training.csv")
train_df = train_df.sample(10000)
for label in REGRESSION_LABELS:
	train_df[label] = train_df[label].apply(lambda x : float(x))
train_df = train_df.dropna()
train_df["stars"] = train_df["stars"].apply(lambda x : int(x) - 1)
train_df.to_csv("data/small_neural_train.csv", index_label=False)
