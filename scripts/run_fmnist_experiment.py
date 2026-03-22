import argparse
import torch
from datetime import datetime
import pandas as pd
from src.models.cnn import SmallCNN
from src.datasets import load_fmnist
from src.multiclass_trainer import Trainer
from src.stopping_rules import ThresholdStoppingRule
from src.utils import save_results_csv, plot_training_validation, ensure_dir

# ---------------------------
# Argument parser
# ---------------------------
parser = argparse.ArgumentParser(description="Run Fashion-MNIST experiment with multiple optimizers")
parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
parser.add_argument("--max_iter", type=int, default=50, help="Maximum iterations per epoch")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--lambda_reg", type=float, default=1e-5, help="Regularization coefficient")
parser.add_argument("--eval_mode", type=str, default="running_avg", choices=["full", "running_avg", "single_batch"])
parser.add_argument("--subset_fraction", type=float, default=0.1, help="Fraction of data for evaluation")
parser.add_argument("--lr_sgd", type=float, default=0.01, help="SGD learning rate")
parser.add_argument("--lr_adam", type=float, default=0.001, help="Adam learning rate")
args = parser.parse_args()

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
# Load Fashion-MNIST
# ---------------------------
train_loader, val_loader = load_fmnist(batch_size=args.batch_size)

# ---------------------------
# Model
# ---------------------------
model = SmallCNN().to(device)
trainer = Trainer(model)

# ---------------------------
# Training settings
# ---------------------------
epochs = args.epochs
lambda_reg = args.lambda_reg
eval_mode = args.eval_mode
max_iter = args.max_iter
subset_fraction = args.subset_fraction

print("Fashion-MNIST Training start:", datetime.now())

# ---------------------------
# Train all methods
# ---------------------------
# RD
model_RD, train_loss_RD, val_loss_RD, train_acc_RD, val_acc_RD, time_rd = trainer.training_RD(
    train_loader, val_loader,
    epochs=epochs,
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
    epochs=epochs,
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
    epochs=epochs,
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
    epochs=epochs,
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

df_all = pd.concat([
    build_df("RD", train_loss_RD, val_loss_RD, train_acc_RD, val_acc_RD, time_rd),
    build_df("SGD", train_loss_sgd, val_loss_sgd, train_acc_sgd, val_acc_sgd, time_sgd),
    build_df("Adam", train_loss_adam, val_loss_adam, train_acc_adam, val_acc_adam, time_adam),
    build_df("KFAC", train_loss_KFAC, val_loss_KFAC, train_acc_KFAC, val_acc_KFAC, time_KFAC),
], ignore_index=True)

ensure_dir("results")
save_results_csv(df_all, "results/fmnist_all_methods.csv")

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
    "KFAC": {"mse": train_loss_KFAC, "val_mse": val_loss_KFAC, "train_acc": train_acc_KFAC, "val_acc": val_acc_KFAC},
}

ensure_dir("results/figures")
plot_training_validation(
    time_dict,
    metrics_dict,
    save_path="results/figures/fmnist_training_plot.png",
    title_prefix="Fashion-MNIST"
)