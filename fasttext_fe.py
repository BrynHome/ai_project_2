from pandas import read_csv, concat, DataFrame
import numpy as np
from gensim.models.fasttext import load_facebook_model
import nltk
from nltk.tokenize import word_tokenize
from argparse import ArgumentParser
from os.path import exists
from os import remove

nltk.download('punkt')

PARSER = ArgumentParser()
PARSER.add_argument("filepath", help="The filepath to the training data.")
PARSER.add_argument("-o", "--output", help="The output filename. Output filename is output.csv by default.", default="data/output.csv")
PARSER.add_argument("-v", "--vector-size", help="The size of the word vector, defaults to 25",dest="vector_size", default="25")

CLASS_LABELS = ["stars", "funny", "useful", "cool"]

COL_TYPES = {"stars": np.float64, "funny": np.float64, "useful": np.float64, "cool": np.float64, "text": str}

def create_ft_word_vectors(df: DataFrame, output: str, word_vector_size: int):
    model = load_facebook_model(f"models/cc.en.{word_vector_size}.bin")
    count = 0

    # 1. For each training sample, convert 'text' into a list of tokens.
    # 2. For each word in the list of tokens, pass the word into the fast text model to get the word vector for that word.
    # 3. Compute the mean of all the word vectors to create a single vector which represents the original text.
    # 4. This word vector becomes the column values for the sample.
    for batch in df:
        print(f"Batch #{count+1}")
        batch.dropna(inplace=True)
        # Handle missing columns
        if 'stars' not in batch.columns:
            batch['stars'] = 0
        if 'useful' not in batch.columns:
            batch['useful'] = 0
        if 'funny' not in batch.columns:
            batch['funny'] = 0
        if 'cool' not in batch.columns:
            batch['cool'] = 0
        if 'text' not in batch.columns:
            batch['text'] = ""
        mean_vectors = []
        for index, sample in batch.iterrows():
            tokens = word_tokenize(sample["text"])
            word_vectors = [model.wv[word] for word in tokens]
            if not word_vectors:
                word_vectors = [[0 for i in range(0, word_vector_size)]]
            mean_vectors.append(np.mean(np.array(word_vectors), axis=0))
        labels = [f"x{i+1}" for i in range(0, len(mean_vectors[0]))]
        mean_vector_dataframe = DataFrame(mean_vectors, columns=labels)
        batch_out = concat([mean_vector_dataframe.reset_index(), batch[["stars", "funny", "useful", "cool"]].reset_index()], axis=1)
        batch_out.drop(["index"], axis=1, inplace=True)
        if count == 0:
            batch_out.to_csv(output, index=False, header=True, mode="a")
        else:
            batch_out.to_csv(output, index=False, header=False, mode="a")
        count += 1

if __name__ == "__main__":

    ARGS = PARSER.parse_args()

    if exists(ARGS.output):
        remove(ARGS.output)

    training_batches = read_csv(ARGS.filepath, chunksize=100000, dtype=COL_TYPES, encoding='utf8')
    
    create_ft_word_vectors(training_batches, ARGS.output, int(ARGS.vector_size))
