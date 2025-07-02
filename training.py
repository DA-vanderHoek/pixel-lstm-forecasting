# training.py

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
from model_architecture import PixelLSTM
from dataloader_utils import PixelTimeSeriesDataset, split_data, load_data

# ------------------- Config ------------------- #
params = {
    "hidden_size": 256,
    "num_layers": 3,
    "dropout": 0.3987832655685518,
    "lr": 0.004187766588366865,
    "batch_size": 32,
    "n_warmup": 0,
    "train_ratio": 0.3953,
    "val_ratio": 0.2793,
}

Downsample = 1.0
max_epochs = 150
early_stopping_patience = 5
save_dir = "trained_output"
os.makedirs(save_dir, exist_ok=True)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X, static, y, coords = load_data()
    splits = split_data(X, static, y, coords, splits=(params["train_ratio"], params["val_ratio"], 1.0 - params["train_ratio"] - params["val_ratio"]), downsample_ratio=Downsample)

    X_train, static_train, y_train, _ = splits['train']
    X_val, static_val, y_val, _ = splits['val']
    X_test, static_test, y_test, coords_test = splits['test']

    train_loader = DataLoader(PixelTimeSeriesDataset(X_train, y_train, static_train), batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(PixelTimeSeriesDataset(X_val, y_val, static_val), batch_size=params["batch_size"])
    test_loader = DataLoader(PixelTimeSeriesDataset(X_test, y_test, static_test), batch_size=params["batch_size"])

    model = PixelLSTM(X.shape[2], static.shape[1], params["hidden_size"], params["num_layers"], params["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    criterion = torch.nn.MSELoss()

    best_val_loss = float('inf')
    best_state_dict = None
    train_losses, val_losses = [], []

    for epoch in range(max_epochs):
        model.train()
        batch_losses = []
        for Xb, yb, xs in train_loader:
            Xb, yb, xs = Xb.to(device), yb.to(device), xs.to(device)
            optimizer.zero_grad()
            out, _ = model(Xb, xs, return_hidden=True)
            if params["n_warmup"] > 0:
                out, yb = out[:, params["n_warmup"]:], yb[:, params["n_warmup"]:]
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(loss.item())

        train_losses.append(np.mean(batch_losses))

        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for Xv, yv, xs in val_loader:
                Xv, xs = Xv.to(device), xs.to(device)
                pred, _ = model(Xv, xs, return_hidden=True)
                val_preds.append(pred.cpu().numpy())
                val_trues.append(yv.numpy())

        val_loss = mean_squared_error(np.concatenate(val_trues).ravel(), np.concatenate(val_preds).ravel())
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_losses[-1]:.4f} | Val MSE: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    # Save model & results
    model_path = os.path.join(save_dir, "best_model.pt")
    torch.save(best_state_dict, model_path)
    print(f"✅ Model saved to: {model_path}")

    pd.DataFrame({"epoch": range(1, len(train_losses)+1), "train_loss": train_losses, "val_loss": val_losses}) \
        .to_csv(os.path.join(save_dir, "loss_log.csv"), index=False)

    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title("Loss Curve")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

    # Final Test
    model.load_state_dict(torch.load(model_path))
    model.eval()
    test_preds, test_trues, test_statics = [], [], []
    with torch.no_grad():
        for Xt, yt, xs in test_loader:
            Xt, yt, xs = Xt.to(device), yt.to(device), xs.to(device)
            pred, _ = model(Xt, xs, return_hidden=True)
            test_preds.append(pred.cpu().numpy())
            test_trues.append(yt.cpu().numpy(_
