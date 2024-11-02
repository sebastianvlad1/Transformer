from config import TRAIN_DATA_PATH
from utils import tokenize
import torch

def load_and_prepare_data():
    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    vocabulary = get_unique_sorted_characters(text)
    stoi = { ch:i for i, ch in enumerate(vocabulary) }
    itos = { i:ch for i, ch in enumerate(vocabulary) }
    tensors = torch.tensor(tokenize(text, stoi), dtype=torch.long)
    return tensors, stoi, itos

def get_unique_sorted_characters(text):
    return sorted(list(set(text)))

def split_data(data, ratio):
    n = int(len(data) * ratio)
    train_data = data[:n]
    test_data = data[n:]
    return train_data, test_data