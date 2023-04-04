from pandas import DataFrame, read_csv, concat, to_numeric
from argparse import ArgumentParser
from collections import Counter

REGRESSION_LABELS = ["funny", "useful", "cool"]
PARSER = ArgumentParser()
PARSER.add_argument("sample", help="Number of entries to sample from the train/test set. Set to 0 for no sample", type=int)
PARSER.add_argument("training", help="Training set to process")
PARSER.add_argument("test", help="Test set to process")
ARGS = PARSER.parse_args()

test = pd.read_csv(ARGS.test)
test.dropna(inplace=True)

train_df = read_csv(ARGS.training)
if ARGS.sample > 0:
	SAMPLE_AMOUNT = ARGS.sample 
	test = test.sample(SAMPLE_AMOUNT)
	train_df = train_df.sample(SAMPLE_AMOUNT)
for label in REGRESSION_LABELS:
	train_df[label] = train_df[label].apply(lambda x : float(x))
train_df = train_df.dropna()
train_df["stars"] = train_df["stars"].apply(lambda x : int(x) - 1)
word_count = Counter(" ".join(test[""]).split()).most_common(5)
word_freq = DataFrame(word_count, columns=["Words","Freq"])
expr2 = test.replace(word_freq.words.tolist(), "")
expr3_class = test.drop(test[test['stars'] == 0 or test['stars'] == 4].index)
train_df.to_csv("data/small_neural_train.csv", index_label=False)
test.to_csv("data/small_neural_test.csv", index_label=False)
expr2.to_csv("data/bh_experiment2.csv", index_label=False)
expr3_class.to_csv("data/bh_experiment3.csv", index_label=False)

