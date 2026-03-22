import torch


# ---------------------------
# Base Class
# ---------------------------
class BaseStoppingRule:
    def __call__(self, metrics):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__


# ---------------------------
# 1. Threshold-based stopping
# ---------------------------
class ThresholdStoppingRule(BaseStoppingRule):
    def __init__(self, loss_threshold=0.01, acc_threshold=0.95, verbose=True):
        self.loss_threshold = loss_threshold
        self.acc_threshold = acc_threshold
        self.verbose = verbose

    def __call__(self, metrics):
        
        if (metrics["train_loss"] < self.loss_threshold and
            metrics["train_acc"] > self.acc_threshold):
            if self.verbose:
                print(f"Early stopping (Threshold): "
                      f"loss={metrics['train_loss']:.4f}, acc={metrics['train_acc']:.4f}")
            return True
        return False

    def __repr__(self):
        return f"ThresholdStoppingRule(loss<{self.loss_threshold}, acc>{self.acc_threshold})"


# ---------------------------
# 2. Validation loss stopping
# ---------------------------
class ValidationLossStoppingRule(BaseStoppingRule):
    def __init__(self, patience=5, delta=0.0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose

        self.best_loss = float("inf")
        self.counter = 0

    def __call__(self, metrics):
        val_loss = metrics["val_loss"]

        if val_loss < self.best_loss - self.delta:
            if self.verbose:
                print(f"Validation improved: {self.best_loss:.6f} → {val_loss:.6f}")
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement ({self.counter}/{self.patience})")

            if self.counter >= self.patience:
                if self.verbose:
                    print("Early stopping (Validation Loss)")
                return True

        return False

    def __repr__(self):
        return f"ValidationLossStoppingRule(patience={self.patience}, delta={self.delta})"


# ---------------------------
# 3. EMA training loss stopping
# ---------------------------
class EMATrainingStoppingRule(BaseStoppingRule):
    def __init__(self, patience=5, alpha=0.1, verbose=False):
        self.patience = patience
        self.alpha = alpha
        self.verbose = verbose

        self.ema_loss = None
        self.best_ema = float("inf")
        self.counter = 0

    def __call__(self, metrics):
        loss = metrics["train_loss"]

        # Compute EMA
        if self.ema_loss is None:
            self.ema_loss = loss
        else:
            self.ema_loss = self.alpha * loss + (1 - self.alpha) * self.ema_loss

        # Check improvement
        if self.ema_loss < self.best_ema:
            self.best_ema = self.ema_loss
            self.counter = 0
            if self.verbose:
                print(f"EMA improved → {self.ema_loss:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No EMA improvement ({self.counter}/{self.patience})")

            if self.counter >= self.patience:
                if self.verbose:
                    print("Early stopping (EMA Training Loss)")
                return True

        return False

    def __repr__(self):
        return f"EMATrainingStoppingRule(patience={self.patience}, alpha={self.alpha})"


# ---------------------------
# 4. Gradient norm stopping
# ---------------------------
class GradientStoppingRule(BaseStoppingRule):
    def __init__(self, patience=5, grad_norm_threshold=1e-6, verbose=False):
        self.patience = patience
        self.threshold = grad_norm_threshold
        self.verbose = verbose

        self.counter = 0

    def __call__(self, metrics):
        model = metrics["model"]
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2

        grad_norm = total_norm ** 0.5

        if grad_norm < self.threshold:
            self.counter += 1
            if self.verbose:
                print(f"Grad norm low ({grad_norm:.2e}) [{self.counter}/{self.patience}]")
        else:
            self.counter = 0
            if self.verbose:
                print(f"Grad norm ok ({grad_norm:.2e})")

        if self.counter >= self.patience:
            if self.verbose:
                print("⏹ Early stopping (Gradient Norm)")
            return True

        return False

    def __repr__(self):
        return f"GradientStoppingRule(patience={self.patience}, threshold={self.threshold})"


