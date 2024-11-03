from data_preprocessing import load_and_prepare_data, split_data
from utils import get_batch, detokenize
from BigramModel import BigramLanguageModel
from evaluation import estimate_loss
from training_model import train_model
import torch

def main():
    data, stoi, itos, vocabulary = load_and_prepare_data()

    train_data, val_data = split_data(data, 0.9);

    block_size = 8
    batch_size = 4

    model = BigramLanguageModel(len(vocabulary))

    print(detokenize(model.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist(), itos))

    eval_iters = 10000

    train_model(eval_iters, model, train_data, val_data, batch_size, block_size)

    print(detokenize(model.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=400)[0].tolist(), itos))


if __name__ == "__main__":
    main()
