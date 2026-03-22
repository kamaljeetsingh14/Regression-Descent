import torch
import argparse
from datetime import datetime
import pandas as pd

from src.models.cnn import SmallCNN
from src.datasets import load_mnist
from src.multiclass_trainer import Trainer
from src.stopping_rules import (
    ValidationLossStoppingRule,
    ThresholdStoppingRule,
    EMATrainingStoppingRule,
    GradientStoppingRule
)
from src.utils import save_results_csv, plot_training_validation, ensure_dir


# ---------------------------
# Argument Parser
# ---------------------------
def get_args():
    parser = argparse.ArgumentParser(description="MNIST Training Experiment")

    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--max_iter", type=int, default=50,
                        help="Maximum training iterations")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lambda_reg", type=float, default=1e-5,
                        help="Regularization strength (RD)")
    parser.add_argument("--eval_mode", type=str, default="running_avg",
                        choices=["full", "subset", "single_batch", "running_avg"],
                        help="Evaluation strategy")
    parser.add_argument("--subset_fraction", type=float, default=0.1,
                        help="Fraction for subset evaluation")
    parser.add_argument("--lr_sgd", type=float, default=0.01,
                        help="Learning rate for SGD")
    parser.add_argument("--lr_adam", type=float, default=0.001,
                        help="Learning rate for Adam")

    return parser.parse_args()


# ---------------------------
# Main
# ---------------------------
def main():
    args = get_args()

    print("Running with config:")
    print(vars(args))

    # ---------------------------
    # Stopping rule (default)
    # ---------------------------
    stopping_rule = ThresholdStoppingRule(loss_threshold=0.01, acc_threshold=0.95)

    # ---------------------------
    # Setup
    # ---------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = load_mnist(batch_size=args.batch_size)

    model = SmallCNN().to(device)
    trainer = Trainer(model)

    # ---------------------------
    # Training
    # ---------------------------
    print("Training start:", datetime.now())

    # RD
    model_RD, train_loss_RD, val_loss_RD, train_acc_RD, val_acc_RD, time_rd = trainer.training_RD(
        train_loader, val_loader,
        epochs=args.epochs,
        lambdaa=args.lambda_reg,
        adaptive_reg=False,
        max_iter=args.max_iter,
        stopping_rule=stopping_rule,
        eval_mode=args.eval_mode,
        subset_fraction=args.subset_fraction,
        tau=2.7,
        nu=1.8
    )

    # SGD
    model_SGD, train_loss_sgd, val_loss_sgd, train_acc_sgd, val_acc_sgd, time_sgd = trainer.training_SGD(
        train_loader, val_loader,
        epochs=args.epochs,
        optimize=None,
        learning_rate=args.lr_sgd,
        max_iter=args.max_iter,
        stopping_rule=stopping_rule,
        eval_mode=args.eval_mode,
        subset_fraction=args.subset_fraction
    )

    # Adam
    model_ADAM, train_loss_adam, val_loss_adam, train_acc_adam, val_acc_adam, time_adam = trainer.training_SGD(
        train_loader, val_loader,
        epochs=args.epochs,
        optimize="Adam",
        learning_rate=args.lr_adam,
        max_iter=args.max_iter,
        stopping_rule=stopping_rule,
        eval_mode=args.eval_mode,
        subset_fraction=args.subset_fraction
    )

    # KFAC
    model_KFAC, train_loss_KFAC, val_loss_KFAC, train_acc_KFAC, val_acc_KFAC, time_KFAC = trainer.train_KFAC(
        train_loader, val_loader,
        epochs=args.epochs,
        optimize="Adam",
        learning_rate=args.lr_adam,
        max_iter=args.max_iter,
        stopping_rule=stopping_rule,
        use_kfac=True,
        eval_mode=args.eval_mode,
        subset_fraction=args.subset_fraction
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
    save_results_csv(df_all, "results/mnist_all_methods.csv")

    
    # Plot
    
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
        save_path="results/figures/mnist_training_plot.png",
        title_prefix="MNIST"
    )

    



if __name__ == "__main__":
    main()