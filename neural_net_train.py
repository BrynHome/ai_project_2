import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from bh_neural_net_simple import Dataset, Classify
from sklearn.model_selection import train_test_split


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
                torch.save(model, f"best_model.pt")
                print("Saved model")
                early_stopping_threshold_count = 0
            else:
                early_stopping_threshold_count += 1

            if early_stopping_threshold_count >= 1:
                print("Early stopping")
                break


if __name__ == "__main__":
    train_df = pd.read_csv("training.csv")
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    train_dl = DataLoader(Dataset(train_df, tokenizer), batch_size=8, shuffle=True, num_workers=0)
    val_dl = DataLoader(Dataset(val_df, tokenizer), batch_size=8, num_workers=0)
    model = Classify()

    learn_rate = 1e-6
    epochs = 4
    train(model, train_dl, val_dl, learn_rate, epochs)
