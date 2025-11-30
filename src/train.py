import os
import numpy as np
from rnn import RNN
from lstm import LSTM
from preprocess import preprocess_data
import matplotlib.pyplot as plt
import sys

def train(model, X_train, y_train, batch_size=16, epochs=10, verbose=1):
    #function trains model with batching the arrays/data
    print("Starting training.\n")
    all_losses = []
    numSamplesTrained = len(X_train)
    
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        count = 0
        # shuffle the indices
        indices = np.random.permutation(numSamplesTrained)
        
        # Iterate through batches
        for strt in range(0, numSamplesTrained, batch_size):
            end = min(strt + batch_size, numSamplesTrained)
            batch_idxs = indices[strt:end]
            X_batch = X_train[batch_idxs]
            y_batch = y_train[batch_idxs]
            
            # Train each sample in batch
            for i in range(len(X_batch)):
                x_seq = X_batch[i] #    (window_len, features)
                target = y_batch[i]  # (1,)
                
                # perform backward pass and get loss, one training step
                loss = model.backward(x_seq, target)
                total_loss += loss
                count += 1
        
        # computer  loss for epoch
        avg_loss = total_loss / max(1, count)
        all_losses.append(avg_loss)
        
        if epoch % verbose == 0:
            print(f"Epoch {epoch}/{epochs} — avg loss: {avg_loss:.6f}")
    
    return all_losses

# evaluate model on test set
def evaluate(model, X, y, scaler_output):
    model_preds_scaled = []
    
    for i in range(len(X)):
        pred = model.predict(X[i])
        model_preds_scaled.append(pred)
    
    model_preds_scaled = np.array(model_preds_scaled).reshape(-1, 1)
    targetsScaled = y.reshape(-1, 1)
    #inverse scale the predictions and targets
    model_preds = scaler_output.inverse_transform(model_preds_scaled).flatten()
    targets = scaler_output.inverse_transform(targetsScaled).flatten()
    
    mse = np.mean((model_preds - targets)**2)
    mae = np.mean(np.abs(model_preds - targets))
    rmse = np.sqrt(mse)
    print(f"\nMSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}")
    return model_preds, targets, {"mse": mse, "mae": mae, "rmse": rmse}


def plot_losses(losses, plot_path="training_loss.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.savefig(plot_path)
    print(f"Saved loss plot to {plot_path}")
    plt.close()


def plot_predictions(model_preds, targets, title, plot_path):
    plt.figure(figsize=(10, 5))
    plt.plot(targets, label="True", linewidth=2)
    plt.plot(model_preds, label="Predicted", alpha=0.8)
    plt.xlabel("Time")
    plt.ylabel("Actual Price")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved prediction plot to {plot_path}")
    plt.close()


def plot_bar_metric(results, metric, plot_path, title=None):
    names = []
    values = []
    for res in results:
        name = res["name"]
        value = res["metrics"][metric]
        names.append(name)
        values.append(value)
    
    plt.figure(figsize=(8, 5))
    plt.bar(names, values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric.upper())
    if title is None:
        title = f"{metric.upper()} by Model Configuration"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved metric bar plot to {plot_path}")
    plt.close()


def main():
    #create log for experiemtns
    log_file = open("results/training_output.log", "w")
    sys.stdout = log_file
    
    csv_path = "data/raw/raw_data.csv"
    
    # config for hyperparam testng
    numInputs = 5
    numOutput = 1
    window_len = 30
    batch_size = 16
    hidden_sizes = [16, 32]
    learning_rates = [0.001, 0.005]
    epochs_list = [20, 30, 50]
    
    # Load and preprocess data once
    X_train, y_train, X_test, y_test, scaler_input, scaler_output = preprocess_data(
        csv_path,
        window_len=window_len,
        train_test_split=0.8
    )
    
    #calc stuff for datasize
    total_size = len(X_train) + len(X_test)
    train_pct = (len(X_train) / total_size) * 100
    test_pct = (len(X_test) / total_size) * 100
    
    # Train models with different configurations
    for model_type in ["RNN", "LSTM"]:
        results = []
        allLosses = {}
        
        for hidden_size in hidden_sizes:
            for lr in learning_rates:
                for epochs in epochs_list:
                    model_name = f"{model_type}_hs{hidden_size}_lr{lr}_ep{epochs}"
                    print(f"Training Model: {model_name}")
                    print("Parameter Tested:")
                    print(f"\tModel Type = {model_type}")
                    print(f"\tHidden Size = {hidden_size}")
                    print(f"\tLearning Rate = {lr}")
                    print(f"\tEpochs = {epochs}")
                    print(f"\tNumber of Inputs = {numInputs}")
                    print(f"\tNumber of Outputs = {numOutput}")
                    print(f"\tWindow Length = {window_len}")
                    print(f"\tBatch Size = {batch_size}")
                    print(f"\tError Function = MSE/MAE")
                    print("\nDataset Information:")
                    print(f"\tTotal Samples = {total_size}")
                    print(f"\tTrain/Test Split = {int(train_pct)}:{int(test_pct)}")
                    print(f"\tTrain Samples = {len(X_train)}")
                    print(f"\tTest Samples = {len(X_test)}")
                    
                    # Initialize model
                    if model_type == "RNN":
                        model = RNN(numInputs, hidden_size, numOutput, window_len, lr)
                    else:
                        model = LSTM(numInputs, hidden_size, numOutput, window_len, lr)
                    
                    # Train model
                    losses = train(model, X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
                    allLosses[model_name] = losses
                    
                    # Evaluate on test set
                    print("\nTest Evalution:")
                    model_preds, targets, metrics = evaluate(model, X_test, y_test, scaler_output)
                    
                    # Evaluate on train set
                    print("\nTrain Evaluation:")
                    train_preds, train_targets, train_metrics = evaluate(model, X_train, y_train, scaler_output)
                    
                    # Save model
                    model_path = os.path.join("models", f"{model_name}.npz")
                    model.save(model_path)
                    print(f"Saved model to {model_path}")
                    
                    # Save training loss plot
                    plot_losses(losses, plot_path=f"results/{model_name}_loss.png")
                    
                    # Save prediction plot
                    plot_predictions(model_preds,targets,title=f"{model_name} Predictions",plot_path=f"results/{model_name}_predictions.png")
                    
                    # Store results
                    results.append({"name": model_name, "metrics": metrics,"train_metrics": train_metrics })
        
        # Identify best model
        best_model = min(results, key=lambda r: r["metrics"]["mse"])
        print(f"Best {model_type} Model: {best_model['name']}")
        print(f"MSE: {best_model['metrics']['mse']:.6f}")
        print(f"MAE: {best_model['metrics']['mae']:.6f}")
        
        # Plot comparison charts
        plot_bar_metric(
            results, 
            "mse", 
            plot_path=f"results/{model_type}_mse_comparison.png",
            title=f"{model_type} MSE Comparison"
        )
        plot_bar_metric(
            results, 
            "mae", 
            plot_path=f"results/{model_type}_mae_comparison.png",
            title=f"{model_type} MAE Comparison"
        )
    
    sys.stdout = sys.__stdout__
    log_file.close()
    print("\nTraining complete! Best models saved in results/ directory.")


if __name__ == "__main__":
    main()