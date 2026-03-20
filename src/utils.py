import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
import os

# ---------------------------
# Result saving
# ---------------------------
def save_results_csv(df, filepath):
    """
    Save a pandas DataFrame to CSV and create parent folders if missing.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Saved results to {filepath}")


# ---------------------------
# Training & Validation Plotting
# ---------------------------
def plot_training_validation(time_dict, metrics_dict, save_path=None, title_prefix="Training"):
    """
    Plot training or validation metrics over time for multiple optimizers.
    
    Parameters
    ----------
    time_dict : dict
        Dictionary with optimizer name as key and list of timestamps as values.
    metrics_dict : dict
        Dictionary of dictionaries: metrics_dict[optimizer] = {"mse": [...], "val_mse": [...], "train_acc": [...], "val_acc": [...]}
    save_path : str, optional
        If provided, saves figure to this path.
    title_prefix : str
        Prefix for the plot title.
    """
    
    colors = {
        'RD': "#000000",
        'SGD': "#D55E00",
        'Adam': "#009E73",
        'KFAC': "#CC79A7"
    }

    plt.style.use('default')
    fig, ax1 = plt.subplots(figsize=(11,6))

    # Left axis: MSE
    ax1.set_xlabel('Training Time (seconds)', fontsize=14)
    ax1.set_ylabel('MSE', fontsize=14)
    ax1.set_yscale('log')
    ax1.grid(axis='y', alpha=0.5)
    ax1.tick_params(axis='both', labelsize=12)

    for opt, metrics in metrics_dict.items():
        ax1.plot(time_dict[opt], metrics["mse"], '-', color=colors.get(opt, 'gray'), lw=2)

    # Right axis: Accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', fontsize=14)
    ax2.tick_params(axis='y', labelsize=12)
    for opt, metrics in metrics_dict.items():
        ax2.plot(time_dict[opt], metrics["train_acc"], '--', color=colors.get(opt, 'gray'), lw=2)

    # Legends
    optimizer_legend = [Line2D([0],[0], color=colors.get(opt,'gray'), lw=2, label=opt) for opt in metrics_dict.keys()]
    style_legend = [
        Line2D([0],[0], color='gray', lw=2, linestyle='-', label='MSE'),
        Line2D([0],[0], color='gray', lw=2, linestyle='--', label='Accuracy')
    ]
    legend1 = ax1.legend(handles=optimizer_legend, loc='upper left', frameon=False, fontsize=12)
    legend2 = ax1.legend(handles=style_legend, loc='upper right', frameon=False, fontsize=12)
    ax1.add_artist(legend1)

    plt.title(f"{title_prefix}: Error & Accuracy vs Time", fontsize=16)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.show()


# ---------------------------
# General helpers
# ---------------------------
def ensure_dir(path):
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)