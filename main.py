from data_preprocessing import load_and_prepare_data, split_data
from utils import get_batch
from BigramModel import BigramLanguageModel

def main():
    data, stoi, itos, vocabulary = load_and_prepare_data()

    train_data, val_data = split_data(data, 0.9);

    block_size = 8
    batch_size = 4

    xb, yb = get_batch(train_data, batch_size, block_size)
    # print(xb)
    # for batch in range(batch_size):
    #     for block in range(block_size):
    #         context = xb[batch, :block+1]
    #         target = yb[batch, block]
    #         print(f"When the input is {context}, the output is {target}")

    model = BigramLanguageModel(len(vocabulary))
    out, loss = model(xb,yb)
    print(loss)


if __name__ == "__main__":
    main()
