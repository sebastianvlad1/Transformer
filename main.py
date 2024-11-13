from data_preprocessing import load_and_prepare_data, split_data
from utils import get_batch, detokenize
from BigramModel import BigramLanguageModel
from evaluation import estimate_loss
from training_model import train_model
import torch
from torch.nn import functional as F
import torch.nn as nn

def main():

    data, stoi, itos, vocabulary = load_and_prepare_data()

    train_data, val_data = split_data(data, 0.9)

    block_size = 8
    batch_size = 4
    n_embd = 32

    model = BigramLanguageModel(len(vocabulary), n_embd, block_size)

    #print(detokenize(model.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist(), itos))

    # eval_iters = 10000
    #
    # train_model(eval_iters, model, train_data, val_data, batch_size, block_size)

    #print(detokenize(model.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=400)[0].tolist(), itos))

    torch.manual_seed(1337)
    B,T,C=4,8,2
    x = torch.randn(4,8,2)

    head_size = 16
    key = nn.Linear(C, head_size, bias=False)
    query = nn.Linear(C, head_size, bias=False)
    value = nn.Linear(C, head_size, bias=False)
    k = key(x) # B, T, 16
    q = query(x) # B, T, 16
    wei = q @ k.transpose(-2, -1) # results in a (B, T, T)

    tril = torch.tril(torch.ones(T, T))
    #wei = torch.zeros(T, T)
    wei = wei.masked_fill(tril == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    print(x.shape)
    v = value(x)
    out = wei @ v

    print(out)


if __name__ == "__main__":
    main()
