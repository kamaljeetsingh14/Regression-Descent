import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from src.models.cnn import SmallCNN
from src.datasets import load_mnist
from src.multiclass_trainer import Trainer
from src.utils import save_results_csv, ensure_dir

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------
# Settings
# ---------------------------
batch_sizes = [16, 32]#, 64, 128, 256, 512]
max_iter = 10
lambda_reg = 0.01
eval_mode = "running_avg"  # "full", "running_avg", or "single_batch"
subset_fraction = 0.1

# ---------------------------
# Load MNIST once
# ---------------------------
train_loader_full, val_loader = load_mnist(batch_size=32)
train_dataset = train_loader_full.dataset
N = len(train_dataset)

# ---------------------------
# Initialize model/trainer
# ---------------------------
model = SmallCNN().to(device)
trainer = Trainer(model)

# ---------------------------
# Run experiment
# ---------------------------
rd_results = {}

print("MNIST RD batch experiment start:", datetime.now())

for batch_size in batch_sizes:
    print(f"Batch size: {batch_size}")

    steps_per_epoch = int(np.ceil(N / batch_size))
    epochs = int(np.ceil(total_iterations / steps_per_epoch))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    model_RD, train_loss, val_loss, train_acc, val_acc, time_rd = trainer.training_RD(
        train_loader,
        val_loader,
        epochs=epochs,
        lambdaa =lambda_reg,
        adaptive_reg=False,
        max_iter=max_iter,
        stopping_rule=None,
        eval_mode=eval_mode,
        subset_fraction=subset_fraction,
    )

    rd_results[batch_size] = {
        "mse": train_loss,
        "val_mse": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "time": time_rd
    }

# ---------------------------
# Save results
# ---------------------------
df_list = []

for batch_size, res in rd_results.items():
    steps = len(res["mse"])
    df = pd.DataFrame({
        "batch_size": [batch_size] * steps,
        "itr": list(range(steps)),
        "time": res["time"],
        "mse": res["mse"],
        "val_mse": res["val_mse"],
        "train_acc": res["train_acc"],
        "val_acc": res["val_acc"]
    })
    df_list.append(df)

df_all = pd.concat(df_list, ignore_index=True)
save_results_csv(df_all, "results/mnist_rd_batch.csv")

# ---------------------------
# Plotting

# Ensure directory exists
ensure_dir("results/figures")

colors = plt.cm.viridis(np.linspace(0, 1, len(batch_sizes)))

# Create 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: Training MSE
for i, bs in enumerate(batch_sizes):
    axs[0, 0].plot(rd_results[bs]["time"], rd_results[bs]["mse"],
                   label=f"{bs}", color=colors[i])
axs[0, 0].set_xlabel("Time (s)")
axs[0, 0].set_ylabel("Training MSE")
axs[0, 0].set_yscale("log")
axs[0, 0].grid(True)
axs[0, 0].set_title("Training MSE")

# Top-right: Validation MSE
for i, bs in enumerate(batch_sizes):
    axs[0, 1].plot(rd_results[bs]["time"], rd_results[bs]["val_mse"],
                   label=f"{bs}", color=colors[i], linestyle="--")
axs[0, 1].set_xlabel("Time (s)")
axs[0, 1].set_ylabel("Validation MSE")
axs[0, 1].set_yscale("log")
axs[0, 1].grid(True)
axs[0, 1].set_title("Validation MSE")

# Bottom-left: Training Accuracy
for i, bs in enumerate(batch_sizes):
    axs[1, 0].plot(rd_results[bs]["time"], rd_results[bs]["train_acc"],
                   label=f"{bs}", color=colors[i])
axs[1, 0].set_xlabel("Time (s)")
axs[1, 0].set_ylabel("Training Accuracy")
axs[1, 0].grid(True)
axs[1, 0].set_title("Training Accuracy")

# Bottom-right: Validation Accuracy
for i, bs in enumerate(batch_sizes):
    axs[1, 1].plot(rd_results[bs]["time"], rd_results[bs]["val_acc"],
                   label=f"{bs}", color=colors[i], linestyle="--")
axs[1, 1].set_xlabel("Time (s)")
axs[1, 1].set_ylabel("Validation Accuracy")
axs[1, 1].grid(True)
axs[1, 1].set_title("Validation Accuracy")

# Legends (one combined legend)
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, title="Batch Size", loc="upper center", ncol=len(batch_sizes))

plt.suptitle("MNIST RD: Training & Validation Metrics vs Time (Batch Size Comparison)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("results/figures/mnist_rd_all_metrics.png", dpi=300)
plt.show()