import os
import numpy as np
from rnn import RNN
from preprocess import create_dataloaders
import matplotlib.pyplot as plt

# training function using batches from data loader
def train(model, train_loader, epochs=10, verbose=1):
    print("Starting training.\n")
    all_losses = []

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        count = 0

        # iterate over batches
        for X_batch, y_batch in train_loader:
            # convert to numpy arrays
            X_batch = np.array(X_batch)       
            y_batch = np.array(y_batch)

            # train each sample in batch individually
            for i in range(len(X_batch)):
                x_seq = X_batch[i]           # (seq_len, features)
                target = np.array([y_batch[i]])

                # perform backward pass and get loss, one training step
                loss = model.backward(x_seq, target)
                total_loss += loss
                count += 1

        # computer epoch loss
        avg_loss = total_loss / max(1, count)
        all_losses.append(avg_loss)

        if epoch % verbose == 0:
            print(f"Epoch {epoch}/{epochs} — avg loss: {avg_loss:.6f}")

    return all_losses

# evaluate model on test set
def evaluate(model, test_loader):
    preds = []
    targets = []

    for X_batch, y_batch in test_loader:
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)

        for i in range(len(X_batch)):
            pred = model.predict(X_batch[i])
            preds.append(pred)

        targets.extend(y_batch.tolist())

    preds = np.array(preds)
    targets = np.array(targets)

    mse = np.mean((preds - targets)**2)
    print(f"\nTest MSE: {mse:.6f}")

    return preds, targets, mse

# plot and save training loss curve
def plot_losses(losses, save_path="training_loss.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.savefig(save_path)
    print(f"Saved loss plot → {save_path}")


def main():
    parquet_path = "data/raw/raw_data.parquet"

    # load dataloaders
    train_loader, test_loader, scaler = create_dataloaders(
        parquet_path,
        sequence_length=30,
        batch_size=16,
        shuffle=True
    )

    input_size = 5          # number of features
    hidden_size = 32
    output_size = 1
    seq_len = 30
    lr = 0.001

    # initialize model
    model = RNN(
        inputSize=input_size,
        hiddenSize=hidden_size,
        outputSize=output_size,
        length=seq_len,
        learnRate=lr,
        seed=42
    )

    # train
    losses = train(model, train_loader, epochs=10, verbose=1)

    # evaluate
    preds, targets, mse = evaluate(model, test_loader)

    # save model
    os.makedirs("models", exist_ok=True)
    model.save("models/rnn_stock_model.npz")
    print("Saved model → models/rnn_stock_model.npz")

    # save loss plot to file
    plot_losses(losses)


if __name__ == "__main__":
    main()
