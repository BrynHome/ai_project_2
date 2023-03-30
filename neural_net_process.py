from datasets import load_dataset
train_df = load_dataset("csv", data_files="data/training.csv")
train_df = train_df["train"]
for x in train_df["stars"]:
    train_df["stars"][x] = train_df["stars"][x] - 1
train_df = train_df.remove_columns(["Unnamed: 0"])
train_df = train_df.rename_column("stars", "label")
train_df.to_csv("data/neural_train.csv")