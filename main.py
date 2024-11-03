from data_preprocessing import load_and_prepare_data, split_data
from utils import get_batch, detokenize
from BigramModel import BigramLanguageModel
from evaluation import estimate_loss
from training_model import train_model
import torch
from torch.nn import functional as F

def main():

    data, stoi, itos, vocabulary = load_and_prepare_data()

    train_data, val_data = split_data(data, 0.9)

    block_size = 8
    batch_size = 4

    model = BigramLanguageModel(len(vocabulary))

    # print(detokenize(model.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist(), itos))
    #
    # eval_iters = 10000
    #
    # train_model(eval_iters, model, train_data, val_data, batch_size, block_size)
    #
    # print(detokenize(model.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=400)[0].tolist(), itos))

    torch.manual_seed(1337)
    B,T,C=4,8,2
    x = torch.randn(4,8,2)

    xbow = torch.zeros(4,8,2)
    for b in range(B):
        for t in range(T):
            xprev = x[b, :t+1]
            xbow[b,t] = torch.mean(xprev, 0)
    print(f"xbow[0]: {xbow[0]}")

    wei = torch.tril(torch.ones(T, T))
    wei = wei / wei.sum(dim=1, keepdim=True)
    xbow2 = wei @ x

    print(torch.allclose(xbow, xbow2))

    #v3
    tril = torch.tril(torch.ones(T, T))
    wei = torch.zeros(T, T)
    wei = wei.masked_fill(tril == 0, float('-inf'))
    wei = F.softmax(wei, dim=1)
    xbow3 = wei @ x
    print(torch.allclose(xbow2, xbow3))



if __name__ == "__main__":
    main()
