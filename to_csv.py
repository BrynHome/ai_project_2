
from pandas import read_json
from argparse import ArgumentParser
from os.path import exists
from os import remove
from sys import exit


PARSER = ArgumentParser()
PARSER.add_argument("filepath", help="The filepath to the raw JSON dataset.")
PARSER.add_argument("-o","--output", dest="csv_output", default="dataset.csv", help="The filepath to the raw CSV output. Defaults to dataset.csv", required=False)


def convert_json_to_csv(infile: str, outfile: str) -> None:
    try:
        reader = read_json(infile, lines=True, chunksize=300000)
    except FileNotFoundError:
        print(f"Could not open {infile}.  Exiting.")
        exit()
    print("Converting raw JSON dataset to CSV...")
    for chunk in reader:
        try:
            chunk = chunk[["stars", "useful", "funny", "cool", "text"]]
            chunk.to_csv(outfile, mode="a", index=False)
        except KeyError as e:
            print(e)


if __name__ == "__main__":

    ARGS = PARSER.parse_args()

    if exists(ARGS.csv_output):
        remove(ARGS.csv_output)

    convert_json_to_csv(infile=ARGS.filepath, outfile=ARGS.csv_output)
