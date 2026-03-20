import torch
from datetime import datetime
from src.models.cnn import SmallCNN
from src.datasets import load_mnist
from src.multiclass_trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, val_loader = load_mnist(batch_size=32)

model = SmallCNN().to(device)
trainer = Trainer(model)

print("RD start:", datetime.now())
model_RD, mse, mse_val, train_acc, val_acc, time_rd = trainer.training_RD(train_loader, val_loader, 1,0.01, adaptive_reg = True, max_iter = 100, tau = 2.7, nu = 1.8)

# Save results
import pandas as pd
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