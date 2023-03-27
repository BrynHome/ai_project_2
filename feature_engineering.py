from pandas import read_csv, concat, DataFrame
from argparse import ArgumentParser
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from joblib import dump, load
from numpy import uint8, uint16
from os.path import exists
from os import remove


PARSER = ArgumentParser()
PARSER.add_argument("filepath", help="The filepath to the training data.")
PARSER.add_argument("-o", "--output", help="The output filename. Output filename is output.csv by default.", default="output.csv")
PARSER.add_argument("-t", "--train", help="Train and save a count vectorizer and td-idf transformer using the given data.", default=False, action="store_true")


VECTORIZER_PATH = "models/vectorizer.joblib"
TRANSFORMER_PATH = "models/transformer.joblib"


if __name__ == "__main__":

    ARGS = PARSER.parse_args()

    if exists(ARGS.output):
        remove(ARGS.output)

    data = read_csv(ARGS.filepath, dtype={'stars': uint8, 'funny': uint16, 'useful': uint16, 'cool': uint16})
    data.dropna(inplace=True)

    if ARGS.train:

        vectorizer = CountVectorizer(max_features=100, ngram_range=(2, 2))
        transformer = TfidfTransformer()

        print("Fitting CountVectorizer...")
        vectorizer = vectorizer.fit(data["text"])
        
        counts = vectorizer.transform(data["text"])
        print("Fitting TfidfTransformer...")
        transformer = transformer.fit(counts)

        print("Saving objects to disk...")
        dump(transformer, TRANSFORMER_PATH)
        dump(vectorizer, VECTORIZER_PATH)

    else:
        print("Loading objects...")
        vectorizer: CountVectorizer = load(VECTORIZER_PATH)
        transformer: TfidfTransformer = load(TRANSFORMER_PATH)
        print("Transforming data...")
        counts = vectorizer.transform(data["text"])
        tfidf_matrix = transformer.transform(counts)
        labels = transformer.get_feature_names_out()
        print(f"Saving to {ARGS.output}...")        
        data.drop(["text"], axis=1, inplace=True)
        data = concat([data, DataFrame(tfidf_matrix.toarray(), columns=labels)], axis=1)
        data.to_csv(ARGS.output, index=False)
