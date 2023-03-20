"""
Format of record analysis

    stars: 
        output. to be dropped but kept for training.
    
    useful, funny, cool : 
        a large majority of these ratings are default not rated, as
        such, only records that have these values should be used for training.
        This will likely end up in poor accuracy for these, as it is very likely a record
        has no rating of these.

    text: 
        input. could be skimmed or altered to make input easier?
        words like: 'the', 'a' , 'my', 'it', etc.. are likely unimportant so may be removed
        tests with skim and no skim should be done
        punctuation is likely not important
"""
from pandas import DataFrame, read_json, read_csv, concat
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from os.path import exists
from os import remove

PARSER = ArgumentParser()
PARSER.add_argument("filepath", help="The filepath to the raw JSON dataset.")
ARGS = PARSER.parse_args()

RAW_DATASET = "dataset.csv"

if __name__ == "__main__":

    if exists(RAW_DATASET):
        remove(RAW_DATASET)

    print("Converting raw JSON dataset to CSV...")
    reader = read_json(ARGS.filepath, lines=True, chunksize=300000)
    for chunk in reader:
        chunk = chunk[["stars", "useful", "funny", "cool", "text"]]
        chunk.to_csv(RAW_DATASET, mode="a", index=False)

    # I find it a lot easier to
    # process data when it is a 
    # string.
    COLUMN_TYPES = {
        "text": str,
        "useful": str,
        "funny": str,
        "cool": str,
        "stars": str
    }

    CLASS_LABELS = ["stars", "useful", "funny", "cool"]
    full = read_csv(RAW_DATASET, dtype=COLUMN_TYPES)
    test = full[full["text"] != ""] # Get all rows with non-empty text fields.
    print("Splitting dataset into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(full["text"], full[CLASS_LABELS], test_size=0.10, random_state=42)

    # Create the test set.
    print("Saving test set...")
    test: DataFrame = concat([X_test, y_test], axis=1)
    test.to_csv("test.csv", index=False)

    # Create a training sets.
    training = concat([X_train, y_train], axis=1)
    for label in CLASS_LABELS:
        print(f"Creating/preprocessing training set for label: {label}")
        # This seemed like the simplest way to 
        # ensure that the rating column only 
        # contains numbers 1 through 5.
        data = training[
            (training[label] == "1") |
            (training[label] == "2") |
            (training[label] == "3") |
            (training[label] == "4") |
            (training[label] == "5") 
            ]
        data[["text", label]].to_csv(f"training_{label}.csv", index=False)
    print("Done.")
