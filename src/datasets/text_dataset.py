from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> str:
        # WikiText rows have a "text" field containing raw text strings.
        # Some rows are empty or are section headers (e.g. " = Title = ").
        # Filtering of too-short strings happens in the collate function.
        return self.data[idx]["text"]