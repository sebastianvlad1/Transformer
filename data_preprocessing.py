from config import TRAIN_DATA_PATH

def load_and_prepare_data():
    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    text = get_unique_sorted_characters(text)
    return text

def get_unique_sorted_characters(text):
    return sorted(list(set(text)))