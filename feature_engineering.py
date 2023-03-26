from pandas import read_csv, concat, DataFrame
from argparse import ArgumentParser
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from joblib import dump
from numpy import uint8, uint16
from os import exists, remove

PARSER = ArgumentParser()
PARSER.add_argument("filepath", help="The filepath to the training data.")
PARSER.add_argument("-o", "--output", help="The output filename. Output filename is output.csv by default.", default="output.csv")

VECTORIZER_PATH = "./vectorizer.joblib"
TRANSFORMER_PATH = "./transformer.joblib"


if __name__ == "__main__":

    ARGS = PARSER.parse_args()

    if exists(f"data/{ARGS.output}"):
        remove(f"data/{ARGS.output}")

    vectorizer = CountVectorizer(max_features=100)
    transformer = TfidfTransformer()

    training = read_csv(ARGS.filepath, dtype={'stars': uint8, 'funny': uint16, 'useful': uint16, 'cool': uint16})
    training.dropna(inplace=True)

    vectorizer = vectorizer.fit(training["text"])
    counts = vectorizer.transform(training["text"])
    
    transformer = transformer.fit(counts)
    tfidf_matrix = transformer.transform(counts)

    dump(transformer, TRANSFORMER_PATH)

    labels = transformer.get_feature_names_out()

    dump(vectorizer, VECTORIZER_PATH)

    training.drop(["text"], axis=1, inplace=True)
    training = concat([training, DataFrame(tfidf_matrix.toarray(), columns=labels)], axis=1)
    
    training.to_csv(f"data/{ARGS.output}", index=False)
