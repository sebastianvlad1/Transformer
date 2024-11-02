import torch
# Create a character level tokenizer instead of subwords encoding for the simplicity
def tokenize(string, stoi):
    return [stoi[c] for c in string if c in stoi]

# Get a list of integers from the vocabulary and convert it to string
def detokenize(token, itos):
    return ''.join([itos[i] for i in token])

def get_batch(data, batch_size, block_size):
    torch.manual_seed(1337)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y