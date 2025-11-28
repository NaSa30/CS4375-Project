import os
import numpy as np
from rnn import RNN
from lstm import LSTM
from preprocess import create_dataloaders
import matplotlib.pyplot as plt
import sys

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
    mae = np.mean(np.abs(preds - targets))
    # calculate r^2 for accuracy too
    ss_res = np.sum((preds - targets) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 0.0
    if ss_tot == 0:
        r2 = 0.0
    else:
        r2 = 1 - (ss_res / ss_tot)

    print(f"\nAccuracy: R2: {r2:.6f}")
    print(f"\nMSE: {mse:.6f}, MAE:{mae:.6f}")
    return preds, targets, {"mse": mse, "mae": mae, "r2": r2}

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
    plt.close()

def plot_predictions(preds, targets, title, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(targets, label="True", linewidth=2)
    plt.plot(preds, label="Predicted", alpha=0.8)
    plt.xlabel("Time")
    plt.ylabel("Scaled Price")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved prediction plot → {save_path}")
    plt.close()

def plot_bar_metric(results, metric, save_path, title=None):
    names = [r["name"] for r in results]
    values = [r["metrics"][metric] for r in results]

    plt.figure(figsize=(8, 5))
    plt.bar(names, values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric.upper())
    if title is None:
        title = f"{metric.upper()} by Model Configuration"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved metric bar plot → {save_path}")
    plt.close()

def plot_r2_comparison(results, save_path, title=None):
    names = [r["name"] for r in results]
    test_r2 = [r["metrics"]["r2"] for r in results]
    train_r2 = [r["train_metrics"]["r2"] for r in results]
    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, train_r2, width, label="Train R²")
    plt.bar(x + width/2, test_r2, width, label="Test R²")

    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylabel("R² Score")
    if title is None:
        title = "Train vs Test R² Comparison"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved R² comparison plot → {save_path}")
    plt.close()

def main():

    #logging all the console information to track progress and best model 
    log_file = open("results/training_output.log", "w")
    sys.stdout = log_file

    parquet_path = "data/raw/raw_data.parquet"
    # load the dataloaders
    train_loader, test_loader, scaler = create_dataloaders(
        parquet_path,
        sequence_length=30,
        batch_size=16,
        shuffle=True
    )

    # training dataset size info
    train_size = len(train_loader.dataset)
    test_size = len(test_loader.dataset)
    total_size = train_size + test_size
    train_pct = (train_size / total_size) * 100
    test_pct = (test_size / total_size) * 100

    # define hyperparameter grids
    input_size = 5
    output_size  = 1
    seq_len = 30
    hidden_sizes = [16, 32]
    learning_rates = [0.001, 0.005]
    epochs_list = [20, 30, 50]   

    # initialize model by running over all the hyperparameter combinations
    for model_type in ["RNN", "LSTM"]:
        results = [] #this is the results for each model type or configuation
        all_loss_curves = {}  # to store loss curves for all configurations

        for hidden_size in hidden_sizes:
            for lr in learning_rates:
                for epochs in epochs_list:
                    model_name = f"{model_type}_hs{hidden_size}_lr{lr}_ep{epochs}"
                    print(f"\nTraining Model: {model_name}")
                    print("Parameters:")
                    print(f"    Model Type          = {model_type}")
                    print(f"    Hidden Size         = {hidden_size}")
                    print(f"    Learning Rate       = {lr}")
                    print(f"    Epochs              = {epochs}")
                    print(f"    Input Size          = {input_size}")
                    print(f"    Output Size         = {output_size}")
                    print(f"    Sequence Length     = {seq_len}")
                    print(f"    Batch Size          = 16")
                    print(f"    Error Function      = MSE/MAE")
                    print("\nDataset:")
                    print(f"    Total Samples       = {total_size}")
                    print(f"    Train/Test Split    = {int(train_pct)}:{int(test_pct)}")
                    print(f"    Train Samples       = {train_size}")
                    print(f"    Test Samples        = {test_size}")

                    # initialize the model
                    if model_type == "RNN":
                        model = RNN(input_size, hidden_size, output_size, seq_len, lr)
                    else:
                        model = LSTM(input_size, hidden_size, output_size, seq_len,lr)

                    # train the model
                    losses = train(model, train_loader, epochs=epochs, verbose=1)
                    all_loss_curves[model_name] = losses

                    # evaluate the model
                    print("\nTest ")
                    preds, targets, metrics = evaluate(model, test_loader)
                    

                    # evaluate on training set too
                    print("\nTrain ")
                    train_preds, train_targets, train_metrics = evaluate(model, train_loader)
                    # save model
                    model_path = os.path.join("models", f"{model_name}.npz")
                    model.save(model_path)
                    print(f"Saved model → {model_path}")


                    # save training loss plot
                    plot_losses(losses, save_path=f"results/{model_name}_loss.png")

                    # save prediction plot
                    plot_predictions(
                        preds,
                        targets,
                        title=f"{model_name} Predictions",
                        save_path=f"results/{model_name}_predictions.png"
                    )

                    # store results
                    results.append({
                        "name": model_name,
                        "metrics": metrics,
                        "train_metrics": train_metrics
                    })
        best_model = min(results, key=lambda r: r["metrics"]["mse"])
        print(f"\nBest {model_type} Model: {best_model['name']} with MSE: {best_model['metrics']['mse']:.6f}, MAE: {best_model['metrics']['mae']:.6f}")
        
        # plot bar charts for MSE and MAE and R2 comparison
        plot_bar_metric(results, "mse", save_path=f"results/{model_type}_mse_comparison.png", title=f"{model_type} MSE Comparison")
        plot_bar_metric(results, "mae", save_path=f"results/{model_type}_mae_comparison.png", title=f"{model_type} MAE Comparison")
        plot_r2_comparison(results,  save_path=f"results/{model_type}_r2_comparison.png", title=f"{model_type} Train vs Test R² Comparison")



    sys.stdout = sys.__stdout__
    log_file.close()
    print("\nTraining complete! Best models saved in results/ directory.")


if __name__ == "__main__":
    main()
