import torch
from datetime import datetime
from src.models.cnn import SmallCNN
from src.datasets import load_mnist
from src.multiclass_trainer import Trainer
from src.stopping_rules import ValidationLossStoppingRule, ThresholdStoppingRule, EMATrainingStoppingRule, GradientStoppingRule
import pandas as pd
# Pick one
# stopping_rule = ValidationLossStoppingRule(patience=5, delta=0.001, verbose=True)
# stopping_rule = ThresholdStoppingRule(loss_threshold=0.01, acc_threshold=0.95)
# stopping_rule = EMATrainingStoppingRule(patience=5, alpha=0.1)
stopping_rule = GradientStoppingRule(patience=5, grad_norm_threshold=1e-6)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, val_loader = load_mnist(batch_size=32)

model = SmallCNN().to(device)
trainer = Trainer(model)

print("RD start:", datetime.now())
model_RD, mse, mse_val, train_acc, val_acc, time_rd = trainer.training_RD(train_loader, val_loader, 1,0.01, adaptive_reg = False, max_iter = 100, stopping_rule = stopping_rule, tau = 2.7, nu = 1.8)
#model_RD, mse, mse_val, train_acc, val_acc, time_rd = trainer.training_RD( train_loader, val_loader, T,lambdaa, adaptive_reg = False, max_iter = 1000, stopping_rule = None, tau = 2.7, nu = 1.8):
print(mse)
print("RD end:", datetime.now())
# Save results

df = pd.DataFrame({
    "method": ["RD"]*len(mse),
    "itr": list(range(len(mse))),
    "time": time_rd,
    "mse": mse,
    "val_mse": mse_val,
    "train_acc": train_acc,
    "val_acc": val_acc
})
df.to_csv("results/mnist_RD.csv", index=False)