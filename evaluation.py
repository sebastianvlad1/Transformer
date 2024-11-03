import torch
from utils import get_batch

@torch.no_grad()
def estimate_loss(eval_iters, model, train_data, val_data, batch_size, block_size):
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