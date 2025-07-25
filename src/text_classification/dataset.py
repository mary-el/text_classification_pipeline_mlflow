import torch
from torch.utils.data import Dataset



class TextClassificationDataset(Dataset):
    def __init__(self, encodings, labels, classes):
        self.encodings = encodings
        self.labels = labels
        self.classes = classes
        self.label_to_id = {label: i for i, label in enumerate(classes)}

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.label_to_id[self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)
