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
from pandas import DataFrame, read_json, read_csv, concat, to_numeric
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from os.path import exists
from os import remove
from sys import exit
import nltk
from nltk.corpus import stopwords
from langdetect import detect
from numpy import nan
import re

PARSER = ArgumentParser()
PARSER.add_argument("filepath", help="The filepath to the raw JSON dataset.")
PARSER.add_argument("-o","--output", dest="csv_output", default="dataset.csv", help="The filepath to the raw CSV output. Defaults to dataset.csv", required=False)

nltk.download('stopwords')

STOPWORDS = set(stopwords.words("english"))
ASCII_WORD = re.compile("[a-zA-Z_]+")

def detect_language(txt):
  try:
    return detect(txt)
  except:
    return nan

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

if __name__ == "__main__":
    ARGS = PARSER.parse_args()

    if exists(ARGS.csv_output):
        remove(ARGS.csv_output)

    try:
        reader = read_json(ARGS.filepath, lines=True, chunksize=300000)
    except FileNotFoundError:
        print(f"Could not open {ARGS.filepath}.  Exiting.")
        exit()

    print("Converting raw JSON dataset to CSV...")

    for chunk in reader:
        try:
            chunk = chunk[["stars", "useful", "funny", "cool", "text"]]
            chunk.to_csv(ARGS.csv_output, mode="a", index=False)
        except KeyError as e:
            print(e)
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
    try:
        full = read_csv(ARGS.csv_output, dtype=COLUMN_TYPES)
    except FileNotFoundError:
        print(f"Could not open {ARGS.csv_output}.  Exiting.")
        exit()
    full = full[full["text"] != ""] # Get all rows with non-empty text fields.
    print("Splitting dataset into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(full["text"], full[CLASS_LABELS], test_size=0.10, random_state=42)

    # Create the test set.
    print("Saving test set...")
    test: DataFrame = concat([X_test, y_test], axis=1)
    # Remove words with non-ascii characters and stop words.
    test["text"] = test["text"].apply(clean_text)
    test.to_csv("test.csv", index=False)

    # Create a training set.
    print("Saving training set...")
    train: DataFrame = concat([X_train, y_train], axis=1)
    # Remove non-numeric values
    for label in CLASS_LABELS:
        train = train[to_numeric(train[label], errors='coerce').notnull()]
    # I am open to better ways of doing this.
    # but this works. Removes negatives.
    train["cool"] = train["cool"].astype(int)
    train["useful"] = train["useful"].astype(int)
    train["funny"] = train["funny"].astype(int)
    train = train[
        (train["useful"] >= 0) &
        (train["cool"] >= 0) &
        (train["funny"] >= 0)
    ]
    # Remove words with non-ascii characters,
    # convert to lowercase,
    # and remove stop words.
    train["text"] = train["text"].apply(clean_text)
    train.to_csv("training.csv", index=False)
    print("Done.")
