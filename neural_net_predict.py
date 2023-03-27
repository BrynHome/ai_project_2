import torch
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from bh_neural_net_simple import Dataset


def predict(model, data):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = model.to(device)

    results = []
    with torch.no_grad():
        model.eval()
        for data_in, x in tqdm(data):
            attn_mask = data_in["attention_mask"].to(device)
            input_ids = data_in["input_ids"].squeeze(1).to(device)

            out = model(input_ids, attn_mask)
            # Threshold set to 50% for now
            out = (out > 0.5).int()
            results.append(out)
    return torch.cat(results).cpu().detach().numpy()


if __name__ == "__main__":
    model = torch.load("bh_model.pt")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    test_df = pd.read_csv("test.csv")
    test_dl = DataLoader(Dataset(test_df, tokenizer), batch_size=8,
                         shuffle=False, num_workers=0)
    test_df['score'] = predict(model, test_dl)
    test_df.to_csv("nn_results.csv", index=False)