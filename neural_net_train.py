import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pandas import DataFrame
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from bh_neural_net_simple import Dataset, Classify
from sklearn.model_selection import train_test_split
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification
import warnings


def encode(texts, tokenizer):
    l = len(texts)
    in_ids = np.ones((l, 256), dtype='int32')
    atn = np.zeros((l, 256), dtype='int32')
    token_t_ids = np.zeros((l, 256), dtype='int32')

    for t, text in enumerate(texts):
        token = tokenizer.tokenize(text, padding=True, truncation=True)
        encoded_token = tokenizer.convert_tokens_to_ids(token[:(256 - 2)])
        in_len = len(encoded_token) + 2
        if in_len >= 256:
            in_len = 256
        in_ids[t, :in_len] = np.asarray([0] + encoded_token + [2], dtype='int32')
        atn[t, : in_len] = 1

    return {
        'input_word_ids': in_ids,
        'input_mask': atn,
        'input_type_ids': token_t_ids
    }


def model_make(cat):
    m = TFAutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=cat)
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.metrics.SparseCategoricalAccuracy()],
        run_eagerly=True
    )
    return m



if __name__ == "__main__":

    batch_size = 32
    print('Batch size:', batch_size)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    print("Reading dataset...")
    train_df = pd.read_csv("data/training.csv")
    n = len(train_df['stars'])
    n2 = int(n*0.99)
    print(n)
    train_df = train_df.dropna()
    train_df = train_df.drop(np.random.choice(train_df.index, n2, replace=False))
    print(len(train_df['stars']))
    texts = train_df['text'].tolist()
    labels = train_df['stars'].tolist()
    categories = sorted(list(set(labels)))  # set will return the unique different entries
    n_categories = len(categories)
    print("Grabbing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    input = tokenizer(texts, padding=True, truncation=True, return_tensors='tf')
    data = tf.data.Dataset.from_tensor_slices((dict(input), labels))
    val_s = int(0.2*(n-n2))
    val_ds = data.take(val_s).batch(32, drop_remainder=True)
    train_ds = data.skip(val_s).batch(32, drop_remainder=True)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

    print("Making model...")
    model = model_make(n_categories)

    print("Training....")
    h = model.fit(train_ds, epochs=3, verbose=1,
                  validation_data=val_ds)
    model.save('data/bh_model')
