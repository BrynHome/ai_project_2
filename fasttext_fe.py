from pandas import read_csv, concat, DataFrame
import numpy as np
from gensim.models.fasttext import load_facebook_model
from nltk.tokenize import word_tokenize
from argparse import ArgumentParser


PARSER = ArgumentParser()
PARSER.add_argument("filepath", help="The filepath to the training data.")
PARSER.add_argument("-o", "--output", help="The output filename. Output filename is training.csv by default.", default="training.csv")

WORD_VECTOR_SIZE = 25
CLASS_LABELS = ["stars", "funny", "useful", "cool"]

if __name__ == "__main__":

    ARGS = PARSER.parse_args()

    training_batches = read_csv(ARGS.filepath, chunksize=100000, dtype={"stars": np.float64, "funny": np.float64, "useful": np.float64, "cool": np.float64, "text": str}, encoding='utf8')
    model = load_facebook_model("cc.en.25.bin")
    count = 0

    # 1. For each training sample, convert 'text' into a list of tokens.
    # 2. For each word in the list of tokens, pass the word into the fast text model to get the word vector for that word.
    # 3. Compute the mean of all the word vectors to create a single vector which represents the original text.
    # 4. This word vector becomes the column values for the sample.
    for batch in training_batches:
        print(f"Batch #{count+1}")
        batch.dropna(inplace=True)
        mean_vectors = []
        for index, sample in batch.iterrows():
            tokens = word_tokenize(sample["text"])
            word_vectors = [model.wv[word] for word in tokens]
            if not word_vectors:
                word_vectors = [[0 for i in range(0, WORD_VECTOR_SIZE)]]
            mean_vectors.append(np.mean(np.array(word_vectors), axis=0))
        labels = [f"x{i+1}" for i in range(0, len(mean_vectors[0]))]
        mean_vector_dataframe = DataFrame(mean_vectors, columns=labels)
        batch_out = concat([mean_vector_dataframe.reset_index(), batch[["stars", "funny", "useful", "cool"]].reset_index()], axis=1)
        batch_out.drop(["index"], axis=1, inplace=True)
        if count == 0:
            batch_out.to_csv(ARGS.output, index=False, header=True, mode="a")
        else:
            batch_out.to_csv(ARGS.output, index=False, header=False, mode="a")
        count += 1
