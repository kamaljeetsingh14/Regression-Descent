import torch
from datetime import datetime
import pandas as pd

from src.models.cnn import SmallCNN
from src.datasets import load_fmnist
from src.multiclass_trainer import Trainer
from src.stopping_rules import ThresholdStoppingRule
from src.utils import save_results_csv, plot_training_validation

# ---------------------------
# Stopping rule
# ---------------------------
stopping_rule = ThresholdStoppingRule(loss_threshold=0.01, acc_threshold=0.95)
#stopping_rule = ValidationLossStoppingRule(patience=5, delta=0.0)
#stopping_rule = EMATrainingStoppingRule(patience=5, delta=
#stopping_rule = GradientStoppingRule(patience=5, grad_threshold=1e-5)

# ---------------------------
# Setup
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, val_loader = load_fmnist(batch_size=32)

epoch = 1
lambda_reg = 0.00001
eval_mode = "running_avg"  # "full", "running_avg", or "single_batch"
max_iter = 50
subset_fraction = 0.1

model = SmallCNN().to(device)
trainer = Trainer(model)

print("FMNIST Training start:", datetime.now())

# ---------------------------
# Training
# ---------------------------
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

model_SGD, train_loss_sgd, val_loss_sgd, train_acc_sgd, val_acc_sgd, time_sgd = trainer.training_SGD(
    train_loader, val_loader,
    epochs=epoch,
    optimize=None,
    learning_rate=0.01,
    max_iter=max_iter,
    stopping_rule=stopping_rule,
    eval_mode=eval_mode,
    subset_fraction=subset_fraction
)

model_ADAM, train_loss_adam, val_loss_adam, train_acc_adam, val_acc_adam, time_adam = trainer.training_SGD(
    train_loader, val_loader,
    epochs=epoch,
    optimize="Adam",
    learning_rate=0.001,
    max_iter=max_iter,
    stopping_rule=stopping_rule,
    eval_mode=eval_mode,
    subset_fraction=subset_fraction
)

model_KFAC, train_loss_KFAC, val_loss_KFAC, train_acc_KFAC, val_acc_KFAC, time_KFAC = trainer.train_KFAC(
    train_loader, val_loader,
    epochs=epoch,
    optimize="Adam",
    learning_rate=0.001,
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

plot_training_validation(
    time_dict,
    metrics_dict,
    save_path="results/fmnist_training_plot.png",
    title_prefix="Fashion-MNIST"
)

print("FMNIST experiment complete.")