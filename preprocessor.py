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
from pandas import DataFrame, read_csv, concat, to_numeric
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from sys import exit
import nltk
from nltk.corpus import stopwords
from imblearn.under_sampling import RandomUnderSampler
import re
from time import perf_counter

PARSER = ArgumentParser()
PARSER.add_argument("filepath", help="The filepath to the raw CSV dataset.")

TEST_PATH = "data/test.csv"
TRAIN_PATH = "data/training.csv"

STOPWORDS = set(stopwords.words("english"))
ASCII_WORD = re.compile("[a-zA-Z_]+")

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


def regex_tokenize(text: str):
    return ASCII_WORD.findall(text)


def clean_text(text: str):
    # tokenize all ascii words using regex.
    words: list[str] = regex_tokenize(text)
    # Convert all text to lowercase.
    words = [word.lower() for word in words]
    # remove stopwords using nltk.corpus.stopwords
    words = [word for word in words if word not in STOPWORDS]
    return " ".join(words)


def undersample(X, y) -> DataFrame:
    sampler = RandomUnderSampler(random_state=42)
    X_sampled, y_sampled = sampler.fit_resample(X, y)
    return concat([X_sampled, y_sampled], axis=1)


def make_train_test_sets(dataset: DataFrame) -> tuple[DataFrame, DataFrame]:
    X_train, X_test, y_train, y_test = train_test_split(dataset[["text", "funny", "useful", "cool"]], dataset[["stars"]], test_size=0.10, random_state=42)
    train: DataFrame = concat([X_train, y_train], axis=1)
    test: DataFrame = concat([X_test, y_test], axis=1)
    return (train, test)


if __name__ == "__main__":

    ARGS = PARSER.parse_args()
    START = perf_counter()
    nltk.download('stopwords')
    CLASS_LABELS = ["stars", "useful", "funny", "cool"]

    print("Reading dataset...")
    try:
        full = read_csv(ARGS.filepath, dtype=COLUMN_TYPES)
    except FileNotFoundError:
        print(f"Could not open {ARGS.filepath}.  Exiting.")
        exit()

    print("Preprocessing dataset...")
    full = full[full["text"] != ""] # Get all rows with non-empty text fields.
    full.dropna(inplace=True) # Drop all null rows.
    full.drop_duplicates(inplace=True) # Drop all duplicates
    full["text"] = full["text"].apply(clean_text) # Clean all the text data at once.

    # Remove non-numeric values
    for label in CLASS_LABELS:
        full = full[to_numeric(full[label], errors='coerce').notnull()]
    # I am open to better ways of doing this.
    # but this works. Removes negatives.
    full["cool"] = full["cool"].astype(int)
    full["useful"] = full["useful"].astype(int)
    full["funny"] = full["funny"].astype(int)
    full = full[
        (full["useful"] >= 0) &
        (full["cool"] >= 0) &
        (full["funny"] >= 0)
    ]

    print("Splitting preprocessed dataset into training and test sets...")
    train, test = make_train_test_sets(full)

    print("Saving test set...")
    test.to_csv(TEST_PATH, index=False)

    print("Undersampling training set to balance classes...")
    train = undersample(train[["text", "useful", "cool", "funny"]], train[["stars"]])

    print("Saving training set...")
    train.to_csv(TRAIN_PATH, index=False)
    END = perf_counter()
    TOTAL = END-START
    print("Done.")
    print(f"Took {TOTAL/60} minutes")
