from config import TRAIN_DATA_PATH

stoi = {}
itos = {}

def load_and_prepare_data():
    global stoi, itos
    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    text = get_unique_sorted_characters(text)
    stoi = { ch:i for i, ch in enumerate(text) }
    itos = { i:ch for i, ch in enumerate(text) }
    return text

def get_unique_sorted_characters(text):
    return sorted(list(set(text)))

def get_stoi():
    return stoi

def get_itos():
    return itos