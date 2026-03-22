import argparse
import torch
from datetime import datetime
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
from src.models.cnn import SmallCNN
from src.datasets import load_mnist
from src.multiclass_trainer import Trainer
from src.stopping_rules import ThresholdStoppingRule
from src.utils import save_results_csv, plot_training_validation, ensure_dir

# ---------------------------
# Argument parser
# ---------------------------
parser = argparse.ArgumentParser(description="Run MNIST binary experiment with multiple optimizers")
parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
parser.add_argument("--max_iter", type=int, default=50, help="Maximum iterations per epoch")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--lambda_reg", type=float, default=1e-5, help="Regularization coefficient")
parser.add_argument("--eval_mode", type=str, default="running_avg", choices=["full", "running_avg", "single_batch"])
parser.add_argument("--subset_fraction", type=float, default=0.1, help="Fraction of data for evaluation")
parser.add_argument("--lr_sgd", type=float, default=0.01, help="SGD learning rate")
parser.add_argument("--lr_adam", type=float, default=0.001, help="Adam learning rate")
parser.add_argument("--class0", type=int, default=0, help="First class for binary experiment")
parser.add_argument("--class1", type=int, default=1, help="Second class for binary experiment")
args = parser.parse_args()

# ---------------------------
# Binary setup
# ---------------------------
binary_classes = [args.class0, args.class1]

# ---------------------------
# Stopping rule
# ---------------------------
stopping_rule = ThresholdStoppingRule(loss_threshold=0.01, acc_threshold=0.95)

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------
# Load MNIST
# ---------------------------
train_loader_full, val_loader_full = load_mnist(batch_size=args.batch_size)

# Filter for binary classes
def filter_binary(loader, classes=binary_classes):
    X_list, y_list = [], []
    for X, y in loader:
        mask = torch.isin(y.argmax(dim=1), torch.tensor(classes))
        if mask.any():
            X_list.append(X[mask])
            y_list.append(y[mask][:, classes])  # keep only two columns
    X_bin = torch.cat(X_list, dim=0)
    y_bin = torch.cat(y_list, dim=0)
    return torch.utils.data.TensorDataset(X_bin, y_bin)

train_dataset_bin = filter_binary(train_loader_full)
val_dataset_bin = filter_binary(val_loader_full)

train_loader = torch.utils.data.DataLoader(train_dataset_bin, batch_size=args.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset_bin, batch_size=args.batch_size, shuffle=False)

# ---------------------------
# Model
# ---------------------------
class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8 * 13 * 13, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 8 * 13 * 13)
        x = self.fc1(x)
        return x

model = SmallCNN().to(device)
trainer = Trainer(model)

# ---------------------------
# Training settings
# ---------------------------
epoch = args.epochs
lambda_reg = args.lambda_reg
eval_mode = args.eval_mode
max_iter = args.max_iter
subset_fraction = args.subset_fraction

# ---------------------------
# Train all methods
# ---------------------------
print("Training start (binary MNIST):", datetime.now())

# RD
model_RD, train_loss_RD, val_loss_RD, train_acc_RD, val_acc_RD, time_rd = trainer.training_RD(
    train_loader, val_loader,
    epochs=epoch,
    lambdaa=lambda_reg,
    adaptive_reg=False,
    max_iter=max_iter,
    stopping_rule=stopping_rule,
    eval_mode=eval_mode,
    subset_fraction=subset_fraction,
    tau=2.7,
    nu=1.8
)

# SGD
model_SGD, train_loss_sgd, val_loss_sgd, train_acc_sgd, val_acc_sgd, time_sgd = trainer.training_SGD(
    train_loader, val_loader,
    epochs=epoch,
    optimize=None,
    learning_rate=args.lr_sgd,
    max_iter=max_iter,
    stopping_rule=stopping_rule,
    eval_mode=eval_mode,
    subset_fraction=subset_fraction
)

# Adam
model_ADAM, train_loss_adam, val_loss_adam, train_acc_adam, val_acc_adam, time_adam = trainer.training_SGD(
    train_loader, val_loader,
    epochs=epoch,
    optimize="Adam",
    learning_rate=args.lr_adam,
    max_iter=max_iter,
    stopping_rule=stopping_rule,
    eval_mode=eval_mode,
    subset_fraction=subset_fraction
)

# KFAC
model_KFAC, train_loss_KFAC, val_loss_KFAC, train_acc_KFAC, val_acc_KFAC, time_KFAC = trainer.train_KFAC(
    train_loader, val_loader,
    epochs=epoch,
    optimize="Adam",
    learning_rate=args.lr_adam,
    max_iter=max_iter,
    stopping_rule=stopping_rule,
    use_kfac=True,
    eval_mode=eval_mode,
    subset_fraction=subset_fraction
)

# ---------------------------
# Save results
# ---------------------------
def build_df(name, mse, mse_val, train_acc, val_acc, time):
    return pd.DataFrame({
        "method": [name] * len(mse),
        "itr": list(range(len(mse))),
        "time": time,
        "mse": mse,
        "val_mse": mse_val,
        "train_acc": train_acc,
        "val_acc": val_acc
    })

df_RD   = build_df("RD", train_loss_RD, val_loss_RD, train_acc_RD, val_acc_RD, time_rd)
df_SGD  = build_df("SGD", train_loss_sgd, val_loss_sgd, train_acc_sgd, val_acc_sgd, time_sgd)
df_ADAM = build_df("Adam", train_loss_adam, val_loss_adam, train_acc_adam, val_acc_adam, time_adam)
df_KFAC = build_df("KFAC", train_loss_KFAC, val_loss_KFAC, train_acc_KFAC, val_acc_KFAC, time_KFAC)

df_all = pd.concat([df_RD, df_SGD, df_ADAM, df_KFAC], ignore_index=True)

ensure_dir("results")
save_results_csv(df_all, "results/mnist_binary_all_methods.csv")

# ---------------------------
# Plot
# ---------------------------
time_dict = {
    "RD": time_rd,
    "SGD": time_sgd,
    "Adam": time_adam,
    "KFAC": time_KFAC
}

metrics_dict = {
    "RD": {"mse": train_loss_RD, "val_mse": val_loss_RD, "train_acc": train_acc_RD, "val_acc": val_acc_RD},
    "SGD": {"mse": train_loss_sgd, "val_mse": val_loss_sgd, "train_acc": train_acc_sgd, "val_acc": val_acc_sgd},
    "Adam": {"mse": train_loss_adam, "val_mse": val_loss_adam, "train_acc": train_acc_adam, "val_acc": val_acc_adam},
    "KFAC": {"mse": train_loss_KFAC, "val_mse": val_loss_KFAC, "train_acc": train_acc_KFAC, "val_acc": val_acc_KFAC}
}

ensure_dir("results/figures")
plot_training_validation(
    time_dict,
    metrics_dict,
    save_path="results/figures/mnist_binary_training_plot.png",
    title_prefix=f"MNIST Binary ({binary_classes[0]} vs {binary_classes[1]})"
)