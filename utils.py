from data_preprocessing import get_stoi, itos, get_itos


# Create a character level tokenizer instead of subwords encoding for the simplicity
def tokenize(string):
    stoi = get_stoi()
    return [stoi[c] for c in string if c in stoi]

# Get a list of integers from the vocabulary and convert it to string
def detokenize(token):
    itos = get_itos()
    return ''.join([itos[i] for i in token])