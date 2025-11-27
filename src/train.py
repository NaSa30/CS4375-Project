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
    print(f"\nTest MSE: {mse:.6f}, MAE:{mae:.6f}")
    return preds, targets, {"mse": mse, "mae": mae}

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

                    # initialize the model
                    if model_type == "RNN":
                        model = RNN(input_size, hidden_size, output_size, seq_len, lr)
                    else:
                        model = LSTM(input_size, hidden_size, output_size, seq_len,lr)

                    # train the model
                    losses = train(model, train_loader, epochs=epochs, verbose=1)
                    all_loss_curves[model_name] = losses

                    # evaluate the model
                    preds, targets, metrics = evaluate(model, test_loader)

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
                        "metrics": metrics
                    })
        best_model = min(results, key=lambda r: r["metrics"]["mse"])
        print(f"\nBest {model_type} Model: {best_model['name']} with MSE: {best_model['metrics']['mse']:.6f}, MAE: {best_model['metrics']['mae']:.6f}")
        
        # plot bar charts for MSE and MAE
        plot_bar_metric(results, "mse", save_path=f"results/{model_type}_mse_comparison.png", title=f"{model_type} MSE Comparison")
        plot_bar_metric(results, "mae", save_path=f"results/{model_type}_mae_comparison.png", title=f"{model_type} MAE Comparison")



    sys.stdout = sys.__stdout__
    log_file.close()
    print("\nTraining complete! Best models saved in results/ directory.")


if __name__ == "__main__":
    main()
