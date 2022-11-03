import torch

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.utterance = data["Utterance"]
        self.target = data["Target"].tolist()
        self.tokenized_sentence = tokenizer(
            self.utterance.tolist(),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=360,
            add_special_tokens=True,
        )

    def __getitem__(self, idx):
        encoded = {
            key: val[idx].clone().detach()
            for key, val in self.tokenized_sentence.items()
        }
        encoded["label"] = torch.tensor(self.target[idx])
        return encoded

    def __len__(self):
        return len(self.label)
