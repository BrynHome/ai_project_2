from pandas import read_csv, DataFrame
from argparse import ArgumentParser

PARSER = ArgumentParser()
PARSER.add_argument("filepath", help="The filepath to the training data.")
PARSER.add_argument("-o", "--output", help="The output filename. Output filename is training.csv by default.", default="training.csv")

ARGS = PARSER.parse_args()

if __name__ == "__main__":
        
    training = read_csv(ARGS.filepath)
    # TODO: tf-idf feature extraction.
    training.to_csv(ARGS.output)
