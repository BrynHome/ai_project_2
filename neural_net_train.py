import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from bh_neural_net_simple import Dataset, Classify
from sklearn.model_selection import train_test_split
import tensorflow
from transformers import TFRobertaModel
import warnings
warnings.filterwarnings("ignore")


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


def model_make(cat, strategy):
    max = 256
    with strategy.scope():
        input_word_ids = tensorflow.keras.Input((max,), name='input_word_ids', dtype=tensorflow.int32)
        input_mask = tensorflow.keras.Input((max,), name='input_mask', dtype=tensorflow.int32)
        input_type_ids = tensorflow.keras.Input((max,), name='input_type_ids', dtype=tensorflow.int32)

        r_model = TFRobertaModel.from_pretrained('roberta-base')
        r_model_2 = r_model(input_word_ids, attention_mask=input_mask, token_type_ids=input_type_ids)
        r_model_2 = r_model_2[0]

        r_model_2 = tensorflow.keras.layers.Dropout(0.1)(r_model_2)
        r_model_2 = tensorflow.keras.layers.Flatten()(r_model_2)
        r_model_2 = tensorflow.keras.layers.Dense(256, 'relu')(r_model_2)
        r_model_2 = tensorflow.keras.layers.Dense(cat, 'softmax')(r_model_2)

        r_model_3 = tensorflow.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=r_model_2)
        r_model_3.compile(tensorflow.keras.optimizers.Adam(1e-5),
                          'sparse_categorical_crossentropy',
                          metrics=['accuracy'])
        return r_model_3


def train(model, train_dl, val_dl, learn_rate, epochs):
    best_loss = float('inf')
    early_stop_count = 0

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    model = model.to(device)
    criterion = criterion.to(device)
    for epoch in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        model.train()
        for train_in, train_label in tqdm(train_dl):
            attn_mask = train_in["attention_mask"].to(device)
            input_ids = train_label["input_ids"].squeeze(1).to(device)
            train_label = train_label.to(device)

            out = model(input_ids, attn_mask)
            loss = criterion(out, train_label.float().unsqueeze(1))

            total_loss_train += loss.item()
            acc = ((out >= 0.5).int() == train_label.unsqueeze(1)).sum()
            total_acc_train += acc

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        with torch.no_grad():
            total_acc_val = 0
            total_loss_val = 0

            model.eval()

            for val_input, val_label in tqdm(val_dl):
                attention_mask = val_input['attention_mask'].to(device)
                input_ids = val_input['input_ids'].squeeze(1).to(device)

                val_label = val_label.to(device)

                output = model(input_ids, attention_mask)

                loss = criterion(output, val_label.float().unsqueeze(1))

                total_loss_val += loss.item()

                acc = ((output >= 0.5).int() == val_label.unsqueeze(1)).sum().item()
                total_acc_val += acc

            print(f'Epochs: {epoch + 1} '
                  f'| Train Loss: {total_loss_train / len(train_dl): .3f} '
                  f'| Train Accuracy: {total_acc_train / (len(train_dl.dataset)): .3f} '
                  f'| Val Loss: {total_loss_val / len(val_dl): .3f} '
                  f'| Val Accuracy: {total_acc_val / len(val_dl.dataset): .3f}')

            if best_val_loss > total_loss_val:
                best_val_loss = total_loss_val
                torch.save(model, f"bh_model.pt")
                print("Saved model")
                early_stopping_threshold_count = 0
            else:
                early_stopping_threshold_count += 1

            if early_stopping_threshold_count >= 1:
                print("Early stopping")
                break


if __name__ == "__main__":
    try:
        # TPU detection. No parameters necessary if TPU_NAME environment variable is set (always set in Kaggle)
        tpu = tensorflow.distribute.cluster_resolver.TPUClusterResolver()
        tensorflow.config.experimental_connect_to_cluster(tpu)
        tensorflow.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tensorflow.distribute.experimental.TPUStrategy(tpu)
        print('Running on TPU ', tpu.master())
    except ValueError:
        # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
        strategy = tensorflow.distribute.get_strategy()

    print('Number of replicas:', strategy.num_replicas_in_sync)
    print("Reading dataset...")
    train_df = pd.read_csv("data/training.csv")
    n = len(train_df['stars'].unique())
    train_df = train_df.dropna()
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    x_train = train_df[['text']].values.reshape(-1)
    y_train = train_df[['stars']].values.reshape(-1)
    x_val = val_df[['text']].values.reshape(-1)
    y_val = val_df[['stars']].values.reshape(-1)
    print("Grabbing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    print("Creating Dataloaders...")
    x_train = encode(x_train, tokenizer)
    x_val = encode(x_val, tokenizer)
    y_train = np.asarray(y_train, dtype='int32')
    y_val = np.asarray(y_val, dtype='int32')

    print("Making model...")
    model = model_make(n, strategy)

    print("Training....")
    with strategy.scope():
        h = model.fit(x_train, y_train, 8 * strategy.num_replicas_in_sync, epochs=3, verbose=1,
                      validation_data=(x_val, y_val))
        model.save('data/bh_model')
