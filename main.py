from data_preprocessing import load_and_prepare_data, split_data
from utils import get_batch, detokenize
from BigramModel import BigramLanguageModel
import torch

def main():
    data, stoi, itos, vocabulary = load_and_prepare_data()

    train_data, val_data = split_data(data, 0.9);

    block_size = 8
    batch_size = 4

    model = BigramLanguageModel(len(vocabulary))

    print(detokenize(model.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist(), itos))

    @torch.no_grad()
    def estimate_loss(eval_iters):
        out = {}
        model.eval()

        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(train_data, batch_size, block_size)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        train_losses = losses.mean()

        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(val_data, batch_size, block_size)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        val_losses = losses.mean()

        model.train()

        return train_losses, val_losses

    eval_iters = 10000
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print(detokenize(model.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=400)[0].tolist(), itos))

    for iter in range(eval_iters):

        if iter % 1000 == 0:
            train_losses, val_losses = estimate_loss(eval_iters)
            print(f"Step {iter}: train loss: {train_losses:.4f}, val loss: {val_losses:.4f}")

        xb,yb = get_batch(train_data, batch_size, block_size)

        logits, loss = model(xb,yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(detokenize(model.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=400)[0].tolist(), itos))


if __name__ == "__main__":
    main()
