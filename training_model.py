import torch
from evaluation import estimate_loss
from utils import get_batch

def train_model(eval_iters, model, train_data, val_data, batch_size, block_size):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for iter in range(eval_iters):

        if iter % 1000 == 0:
            train_losses, val_losses = estimate_loss(eval_iters, model, train_data, val_data, batch_size, block_size)
            print(f"Step {iter}: train loss: {train_losses:.4f}, val loss: {val_losses:.4f}")

        xb, yb = get_batch(train_data, batch_size, block_size)

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()