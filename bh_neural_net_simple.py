import torch.nn as nn
import torch
from transformers import RobertaModel


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, seq_len=512):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        texts = data.text.values.tolist()
        # add preprocess here?
        self.texts = [self.tokenizer(text,
                                     max_length=self.seq_len,
                                     padding="max_length",
                                     truncation=True,
                                     return_tensors="pt")
                      for text in texts]
        labels = data.score.values.tolist()
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label


class Classify(nn.Module):
    def __init__(self):
        super().__init__()
        self.r_model = RobertaModel.from_pretrained('roberta-base')
        # may need to change in features
        self.fc1 = nn.Linear(in_features=768, out_features=1)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, in_ids, attention_mask):
        """
        Passed input ids and attention mask from tokenization
        raw is the CLS token of the last hidden layer
        """
        raw = self.r_model(in_ids, attention_mask)[0][:, 0]
        out = self.fc1(raw)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.sig(out)
        return out
