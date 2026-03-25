# Regression-Descent
This repository contains the implementation and experimental code for the paper: Regression Descent: A Statistical Framework for Neural Network Optimization


## Abstract

We present Regression Descent (RD), a novel optimization algorithm for training deep neural networks that reformulates each gradient step as a regression problem in the span of
the Jacobian. By leveraging the implicit function theorem in overparameterized settings where the number of parameters exceed observations \((p > n)\), we project the optimization onto an \(n\)-dimensional
subspace, enabling the use of statistical techniques and potentially improved conditioning. Our key insight is that in the overparameterized regime, meaningful parameter updates lie in the
row space of the Jacobian matrix, allowing us to solve a lower-dimensional regression problem with explicit regularization control. We establish convergence guarantees for RD under standard smoothness assumptions, showing that it achieves a convergence rate of \(O(1/k)\) for smooth non-convex objectives. Furthermore, we prove that RD exhibits local linear convergence in neighborhoods of strict local minima, with the convergence rate dependent on the condition number of the regularized
Gram matrix. The algorithm naturally handles the ill-conditioning common in neural network optimization through adaptive regularization and extends seamlessly to multi-output problems and mini-batch settings.
Experimental results on Lorenz96, MNIST, and FMNIST demonstrate that RD achieves
up to 40\% faster convergence compared to SGD and Adam in terms of wall-clock time, with strong performance in the presence of activation function saturation. The computational overhead of solving \(m \times m\) linear systems (where \(m\) is the batch size) is offset by
improved convergence properties and GPU-efficient operations. Our work opens new avenues
for understanding neural network optimization through the lens of statistical regression, providing a practical algorithm for scenarios where standard gradient methods struggle.


## Acknowledgements



## Repository Overview

- `src/` – Core code and implementation of the method
- `scripts/` – Scripts for training, evaluation, and experiments
- `data/` – Instructions or sample data for reproducing results
- `results/` – Figures, tables, and outputs from experiments
- `requirements.txt` – Python dependencies

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. To run an experiment, use:

For default setting:
```bash
python -m scripts.run_mnist_experiment
```

For custom setting:
```bash
python -m scripts.run_mnist_experiment \
    --epochs 5 \
    --max_iter 200 \
    --batch_size 64 \
    --lambda_reg 1e-5 \
    --eval_mode running_avg \
    --subset_fraction 0.1 \
    --lr_sgd 0.01 \
    --lr_adam 0.001
```