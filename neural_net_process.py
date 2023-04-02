from pandas import DataFrame, read_csv, concat, to_numeric
from argparse import ArgumentParser

REGRESSION_LABELS = ["funny", "useful", "cool"]
PARSER = ArgumentParser()
PARSER.add_argument("sample", help="Number of entries to sample from the train set.", type=int)
ARGS = PARSER.parse_args()
SAMPLE_AMOUNT = ARGS.sample

train_df = read_csv("data/training.csv")
train_df = train_df.sample(SAMPLE_AMOUNT)
for label in REGRESSION_LABELS:
	train_df[label] = train_df[label].apply(lambda x : float(x))
train_df = train_df.dropna()
train_df["stars"] = train_df["stars"].apply(lambda x : int(x) - 1)
train_df.to_csv("data/small_neural_train.csv", index_label=False)
