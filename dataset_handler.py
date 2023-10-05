from torch.utils.data import Dataset
from transformers import BertTokenizer
from utils import get_kmer_sequence
import pandas as pd


class StandardDataset(Dataset):
    def __init__(self, path:str, dnabert_path: str, max_length: int, kmer_val: int):
        super().__init__()
        self.dataset_df = pd.read_csv(path)
        self.tokenizer = BertTokenizer.from_pretrained(dnabert_path)
        self.max_length = max_length
        self.kmer_val = kmer_val

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, idx):
        seq = get_kmer_sequence(self.dataset_df["sequence"][idx], kmer_val=self.kmer_val)
        label = self.dataset_df["value"][idx]
        seq_id = self.dataset_df["id"][idx]
        seq = self.tokenizer(seq, max_length=self.max_length, padding="max_length", return_tensors='pt', truncation=True,
                                  add_special_tokens=True,
                                  return_attention_mask=True)
        return seq, label, seq_id