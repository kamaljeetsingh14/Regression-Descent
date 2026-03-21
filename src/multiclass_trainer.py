# Suppress specific warnings
import warnings
warnings.filterwarnings(
    "ignore", 
    category=UserWarning, 
    message="We've integrated functorch into PyTorch.*"
)

# Core libraries
import time
from copy import deepcopy
from itertools import tee
from collections import deque

# Numerical & ML libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functorch import jacrev, make_functional, vmap
from torch.nn.utils.convert_parameters import *
from kfac.preconditioner import KFACPreconditioner

if torch.cuda.is_available():
    device = torch.device("cuda")  # CUDA device object
    print(f"GPU: {torch.cuda.get_device_name(0)}")  # Print GPU name
else:
    device = torch.device("cpu")  # Fallback to CPU if GPU is not available
    print("CUDA is not available. Training on CPU...")


class ThresholdStoppingRule:
    def __init__(self, loss_threshold=0.01, acc_threshold=0.95):
        self.loss_threshold = loss_threshold
        self.acc_threshold = acc_threshold
        self.early_stop = False

    def __call__(self, metrics):
        if (metrics["train_loss"] < self.loss_threshold and 
            metrics["train_acc"] > self.acc_threshold):
            self.early_stop = True
            print("Early stopping triggered (threshold rule).")
        return self.early_stop
    
class Error_Stopping_Rule:
    def __init__(self, patience=5, verbose=False, delta=0.005):
        """
        :param patience: How many epochs to wait after the last time validation loss improved.
        :param verbose: If True, prints a message for each validation loss improvement.
        :param delta: Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_loss = torch.inf
        self.counter = 0
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).')
            # Save the best model weights
            self.best_model_wts = model.state_dict()
        else:
            self.counter += 1
            if self.verbose:
                print(f'Validation loss did not improve. Counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True


class TrainingErrorStoppingRule:
    def __init__(self, patience=5, alpha=0.1, verbose=False):
        """
        :param patience: How many consecutive steps to wait without improvement.
        :param alpha: EMA smoothing factor. Smaller = smoother.
        :param verbose: If True, prints status messages.
        """
        self.patience = patience
        self.alpha = alpha
        self.verbose = verbose

        self.ema_loss = None
        self.best_ema_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_loss):
        # Compute EMA
        if self.ema_loss is None:
            self.ema_loss = current_loss
        else:
            self.ema_loss = self.alpha * current_loss + (1 - self.alpha) * self.ema_loss

        # Check for improvement
        if self.ema_loss < self.best_ema_loss:
            self.best_ema_loss = self.ema_loss
            self.counter = 0
            if self.verbose:
                print(f"EMA improved to {self.ema_loss:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement in EMA. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

class Gradient_Stopping_Rule:
    def __init__(self, patience=5, verbose=False, grad_norm_threshold=1e-6):
        """
        :param patience: How many steps to wait after the last time the gradient norm was below the threshold.
        :param verbose: If True, prints a message for each gradient norm below the threshold.
        :param grad_norm_threshold: The threshold below which the gradient norm is considered very small.
        """
        self.patience = patience
        self.verbose = verbose
        self.grad_norm_threshold = grad_norm_threshold
        self.counter = 0
        self.early_stop = False

    def __call__(self, model):
        """
        Checks the gradient norm of the model's parameters and updates the early stopping state.
        :param model: The PyTorch model whose gradients are to be checked.
        """
        grad_norm = self.compute_grad_norm(model)
        
        if grad_norm < self.grad_norm_threshold:
            self.counter += 1
            if self.verbose:
                print(f'Gradient norm ({grad_norm:.6e}) is below the threshold. Counter: {self.counter}/{self.patience}')
        else:
            self.counter = 0
            if self.verbose:
                print(f'Gradient norm ({grad_norm:.6e}) is above the threshold.')
        
        if self.counter >= self.patience:
            self.early_stop = True

    def compute_grad_norm(self, model):
        """
        Computes the L2 norm of all gradients in the model.
        :param model: The PyTorch model whose gradients are to be computed.
        :return: The L2 norm of all gradients.
        """
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5


class Trainer:
    def __init__(self, model):
        self.model = model.to(device)
        self.funcnet, self.parameters = make_functional(self.model)
        #self.funcnet, self.parameters, self.buffers = make_functional_with_buffers(self.model)

    def get_original_shapes(self):
        original_shapes = []
        for _ , param in self.model.named_parameters():
            original_shapes.append(param.shape)
        return original_shapes



    def vector_to_original_tensor(self, original_shapes, vector):
        index = 0
        reshaped_params = []
        for shape in original_shapes:
            size = torch.tensor(shape).prod().item() # extracting the number of points
            param = vector[index:index + size].view(shape) # reshaping into tensor
            reshaped_params.append(param)
            index += size
        return tuple(reshaped_params)
    
    def layers_update(self, original_shapes, vector):
        index = 0
        params = []
        for shape in original_shapes:
            size = torch.tensor(shape).prod().item() # extracting the number of points
            param = vector[index:index + size] # reshaping into tensor
            params.append(param)
            index += size
        return params

    def funcnet_single(self, parameters, x):
        return self.funcnet(parameters, x.unsqueeze(dim=0)).squeeze(dim=0)

    def batched_jacobian(self, parameters, x):
        jacobian = vmap(jacrev(self.funcnet_single), in_dims=(None, 0))(parameters, x)
        jacobian = [j.flatten(start_dim=2) for j in jacobian]
        return torch.cat(jacobian, dim=2)  # Batch x Class x Weight
        
    
    def training_RD(self, train_loader, val_loader, T,lambdaa, adaptive_reg = False, max_iter = 1000, tau = 2.7, nu = 1.8):
        self.original_shapes = self.get_original_shapes()
        model_ibr = deepcopy(self.model)
        parameters = deepcopy(self.parameters)
        criterion = nn.MSELoss()
        stopping_rule = TrainingErrorStoppingRule(patience=5,  alpha=0.1, verbose=False)
        start_time = time.time()
        residuals = deque(maxlen=2)  # Store R_k and R_{k+1} for adaptive regularization
        flag = True
        train_loss, val_loss, train_acc, val_acc, times, lm = [], [], [], [], [], []
        itr = 0
        if adaptive_reg:
            print("training using adaptive regularization")
            for k in range(T):
                data_iter1, data_iter2 = tee(train_loader)

                # Get the first batch from data_iter2 for wrap-around
                first_batch = next(data_iter2)

                # Convert data_iter2 (which is now one step ahead) to a list and append the first batch
                data_iter2 = list(data_iter2)
                data_iter2.append(first_batch)   
                
                
                for (X_batch, y_batch), (X_next, y_next) in zip(data_iter1, data_iter2):
                    lm.append(lambdaa)
                    #print("itr",itr)
                    X_batch,y_batch = X_batch.to(device),y_batch.to(device)
                    D_k = self.batched_jacobian(parameters, X_batch).detach()
                    #D_k = self.batched_jacobian_chunked(parameters, X_batch, chunk_size=8).detach() # trying for resnet18
                    n, C, p = D_k.shape
                    D_k = D_k.permute(1, 0, 2)
                    D_k_prod = torch.bmm(D_k, D_k.transpose(1, 2))  # [C, n, n]
                    G = D_k_prod.permute(1, 2, 0)
                    if flag:
                        R_k = y_batch - model_ibr(X_batch) #torch.func.functional_call(model, dict(parameters), X)  # calculate R_k
                        R_k = R_k.detach()
                        residuals.append(R_k)
                    I = lambdaa*torch.eye(n, device=D_k.device).unsqueeze(2).expand(n, n, C)    # [n, n, C]
                    G_new = G + I  # [1, 3, 3]  broadcasted
                    G_batch = G_new.permute(2, 0, 1)  # [C, n, n] → suitable for batch inversion
                    G_inv_batch = torch.linalg.inv(G_batch)  # Inverts each [n, n] matrix
                    G_inv = G_inv_batch.permute(1, 2, 0)  # [n, n, C]
                    G_batch = G_inv.permute(2, 0, 1)    # [C, n, n]
                    R_batch = R_k.permute(1, 0).unsqueeze(2)  # [C, n, 1]
                    gamma = torch.bmm(G_batch, R_batch)  # [C, n, 1]
                    gamma = gamma.squeeze(2).permute(1, 0)  # [n, C]
                    gamma_batch = gamma.permute(1, 0).unsqueeze(2)  # [C, n, 1]
                    D_k_T = D_k.transpose(1, 2)  # [C, p, n]
                    result = torch.bmm(D_k_T, gamma_batch)  # [C, p, 1]
                    beta = result.squeeze(2)
                    mean_beta = beta.mean(dim=0)  # Shape: [p]

                    stopping_rule(criterion(model_ibr(X_batch), y_batch).item())

                    
                    param_vec = parameters_to_vector(parameters).detach().clone().requires_grad_(True)    # creating vector of params
                    param_vec_old = param_vec
                    param_new = param_vec +  mean_beta.squeeze(-1)
                    parameters = self.vector_to_original_tensor(self.original_shapes,param_new)
                    
                    with torch.no_grad():
                        for src_param, tgt_param in zip(parameters, model_ibr.parameters()):
                            tgt_param.copy_(src_param)
                
                    
                    


                    if itr > 1:
                        print("itr",itr)

                        R_k_PLUS_1 = y_next.to(device) - model_ibr(X_next.to(device)) #torch.func.functional_call(model, dict(parameters), X)  # calculate R_k
                    
                        R_k_PLUS_1 = R_k_PLUS_1.detach()
                        residuals.append(R_k_PLUS_1)
                        
                        numerator = torch.sum(residuals[1] ** 2) / residuals[1].shape[0]
                        denominator = torch.sum(residuals[0] ** 2) / residuals[0].shape[0]
                        rho = numerator / denominator if denominator > 0 else 0.0
                        #print("rho",rho.item())
                        if residuals[0].numel() == residuals[1].numel():
                            A = residuals[1] - residuals[0]   # R_k1 = R_{k+1}, R_k = R_k
                            numerator = torch.sum(A * residuals[0])
                            denominator = torch.sum(A * A)
                            theta_star = -numerator / denominator if denominator > 0 else 0.0
                            theta_star = theta_star.item()
                            
                        else:
                            # if there is a mismatch in the number of samples in the two batches, we cannot compute theta_star reliably
                            # this happens in the last batch of each epoch when the number of samples is not divisible by the batch size
                            # in this case, we set theta_star to 1 to effectively ignore the extrapolation step and just use the new update
                            theta_star = 1
                        if rho <=1.1:
                            flag = False
                            R_k = R_k_PLUS_1
                                
                            del D_k, X_batch, y_batch, beta
                            torch.cuda.empty_cache()
                            
                            if rho < 0.8 and theta_star > 0.9:
                                #print("here")
                                
                                lambdaa = lambdaa/tau
                            
                            if rho > 0.9 and theta_star < 0.5:
                                #print("there")    
                                lambdaa = lambdaa*nu
                            
                            tr = self.evaluate(model_ibr,train_loader,device)
                            tes = self.evaluate(model_ibr,val_loader,device)

                            train_loss.append(tr[0])
                            val_loss.append(tes[0])  
                            train_acc.append(tr[1])
                            val_acc.append(tes[1])
                            times.append(time.time() - start_time)
                                
                        elif rho > 1.1:
                            flag = True
                            
                            parameters = self.vector_to_original_tensor(self.original_shapes,param_vec_old)
                            
                            with torch.no_grad():
                                for src_param, tgt_param in zip(parameters, model_ibr.parameters()):
                                    tgt_param.copy_(src_param)
                            
                            

                            # if rho > 1.1, we reject the update and increase regularization to make the next step more conservative
                            # we use the same gradienst information (D_k and R_k) to compute the next update
                                        
                            lambdaa = lambdaa*nu

                            I = lambdaa*torch.eye(n, device=D_k.device).unsqueeze(2).expand(n, n, C)    # [n, n, C]
                            G_new = G + I  # [1, 3, 3]  broadcasted
                            G_batch = G_new.permute(2, 0, 1)  # [C, n, n] → suitable for batch inversion
                            G_inv_batch = torch.linalg.inv(G_batch)  # Inverts each [n, n] matrix
                            G_inv = G_inv_batch.permute(1, 2, 0)  # [n, n, C]
                            G_batch = G_inv.permute(2, 0, 1)    # [C, n, n]
                            R_batch = R_k.permute(1, 0).unsqueeze(2)  # [C, n, 1]
                            gamma = torch.bmm(G_batch, R_batch)  # [C, n, 1]
                            gamma = gamma.squeeze(2).permute(1, 0)  # [n, C]
                            gamma_batch = gamma.permute(1, 0).unsqueeze(2)  # [C, n, 1]
                            D_k_T = D_k.transpose(1, 2)  # [C, p, n]
                            result = torch.bmm(D_k_T, gamma_batch)  # [C, p, 1]
                            beta = result.squeeze(2)
                            mean_beta = beta.mean(dim=0)  # Shape: [p]

                            

                            del D_k, X_batch, y_batch, beta
                            torch.cuda.empty_cache()

                            param_vec = parameters_to_vector(parameters).detach().clone().requires_grad_(True)    # creating vector of params
                            param_vec_old = param_vec
                            param_new = param_vec +  mean_beta.squeeze(-1)
                            parameters = self.vector_to_original_tensor(self.original_shapes,param_new)
                            
                            with torch.no_grad():
                                for src_param, tgt_param in zip(parameters, model_ibr.parameters()):
                                    tgt_param.copy_(src_param)

                            tr = self.evaluate(model_ibr,train_loader,device)
                            tes = self.evaluate(model_ibr,val_loader,device)

                            train_loss.append(tr[0])
                            val_loss.append(tes[0])  
                            train_acc.append(tr[1])
                            val_acc.append(tes[1])
                            times.append(time.time() - start_time)
                            
                        
                                
                                
                        lambdaa = torch.clamp(torch.tensor(lambdaa), min=1e-8, max=10).item()
                    
                    if itr > max_iter:
                        break
                    # if MSE[-1] < 0.01: #and train_acc[-1] > 0.95:
                    #     print('Early stopping triggered.')
                    #     break
                    itr+=1
                    
                    
                    
                    
                #lambdaa = lambdaa*(k+2)  
                # if MSE[-1] < 0.01 and train_acc[-1] > 0.95:
                #     break
                
            return model_ibr,train_loss, val_loss,train_acc,val_acc, times
    
        else:
            print("training using constant regularization")
            for k in range(T):
                for X_batch, y_batch in train_loader:
                    X_batch,y_batch = X_batch.to(device),y_batch.to(device)
                    D_k = self.batched_jacobian(parameters, X_batch).detach()
                    n, C, p = D_k.shape
                    D_k = D_k.permute(1, 0, 2)
                    D_k_prod = torch.bmm(D_k, D_k.transpose(1, 2))  # [C, n, n]
                    G = D_k_prod.permute(1, 2, 0)
                    R_k = y_batch - model_ibr(X_batch) #torch.func.functional_call(model, dict(parameters), X)  # calculate R_k
                    R_k = R_k.detach()
                    I = lambdaa*torch.eye(n, device=D_k.device).unsqueeze(2).expand(n, n, C)    # [n, n, C]
                    G_new = G + I  # [1, 3, 3]  broadcasted
                    G_batch = G_new.permute(2, 0, 1)  # [C, n, n] → suitable for batch inversion
                    G_inv_batch = torch.linalg.inv(G_batch)  # Inverts each [n, n] matrix
                    G_inv = G_inv_batch.permute(1, 2, 0)  # [n, n, C]
                    G_batch = G_inv.permute(2, 0, 1)    # [C, n, n]
                    R_batch = R_k.permute(1, 0).unsqueeze(2)  # [C, n, 1]
                    gamma = torch.bmm(G_batch, R_batch)  # [C, n, 1]
                    gamma = gamma.squeeze(2).permute(1, 0)  # [n, C]
                    gamma_batch = gamma.permute(1, 0).unsqueeze(2)  # [C, n, 1]
                    D_k_T = D_k.transpose(1, 2)  # [C, p, n]
                    result = torch.bmm(D_k_T, gamma_batch)  # [C, p, 1]
                    beta = result.squeeze(2)
                    mean_beta = beta.mean(dim=0)  # Shape: [p]

                    stopping_rule(criterion(model_ibr(X_batch), y_batch).item())

                    
                    param_vec = parameters_to_vector(parameters).detach().clone().requires_grad_(True)    # creating vector of params
                    param_vec_old = param_vec
                    param_new = param_vec +  mean_beta.squeeze(-1)
                    parameters = self.vector_to_original_tensor(self.original_shapes,param_new)
                    
                    with torch.no_grad():
                        for src_param, tgt_param in zip(parameters, model_ibr.parameters()):
                            tgt_param.copy_(src_param)
                
                    tr = self.evaluate(model_ibr,train_loader,device)
                    tes = self.evaluate(model_ibr,val_loader,device)

                    train_loss.append(tr[0])
                    val_loss.append(tes[0])  
                    train_acc.append(tr[1])
                    val_acc.append(tes[1])
                    times.append(time.time() - start_time)

                    metrics = {
                        "train_loss": None,
                        "val_loss": val_loss,
                        "train_acc": None,
                        "val_acc": val_acc,
                        "model": model_ibr
                    }
                        
                    if stopping_rule is not None:
                        if stopping_rule(metrics):
                            print(f"Early stopping at iteration {itr}")
                            break

                    if itr > max_iter:
                        break

                    itr += 1
                    
                    
                    print("itr ========",itr)
                #lambdaa = lambdaa*(k+2)  
                # if MSE[-1] < 0.01 and train_acc[-1] > 0.95:
                #     break

                
            return model_ibr,train_loss, val_loss,train_acc,val_acc, times
   
 
    
    def training_SGD(self, train_dataloader, test_dataloader, epochs, optimize ,learning_rate, testing):
        model = deepcopy(self.model)
        criterion = nn.MSELoss()
        if optimize == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            print("training using ADAM")
        else:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            print("training using plain SGD")
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

        
        MSE = []
        MSE_val = []
        train_acc = []
        val_acc = []
        itr=0
        stopping_rule = TrainingErrorStoppingRule(patience=5,  alpha=0.1, verbose=False)
        start_time = time.time()  # record training start time
        times = [] 
        for epoch in range(epochs):
            model.train()
            # Training loop
            for X_batch, y_batch in train_dataloader:
                
                
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred.squeeze(dim=1), y_batch.float())
                loss.backward()
                optimizer.step()
                #stopping_rule(criterion(model(X_batch), y_batch).item())

            # Calculate MSE on the full dataset
                tr = self.evaluate(model,train_dataloader,device)
                tes = self.evaluate(model,test_dataloader,device)

                MSE.append(tr[0])
                MSE_val.append(tes[0])  
                train_acc.append(tr[1])
                val_acc.append(tes[1])
                times.append(time.time() - start_time)      
                
                
                if testing and times[-1] > 10000:
                    break
                itr+=1
                print("itr ========",itr)
            
            
        
            #     early_stopping(model)
                
        
                #if MSE[-1] < 0.01 and train_acc[-1] > 0.95:
                    #print('Early stopping triggered.')
                    #break


                
            scheduler.step()
            #if MSE[-1] < 0.01 and train_acc[-1] > 0.95:
                #break
                
        return model, MSE, MSE_val, train_acc, val_acc, times
    
    def train_KFAC(self, train_dataloader, test_dataloader, epochs, optimize ,learning_rate, max_iter, use_kfac=False):
        model = deepcopy(self.model)
        criterion = nn.MSELoss()
        if optimize == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            print("training using ADAM KFAC")
        else:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            print("training using plain SGD KFAC")
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        precond = KFACPreconditioner(model) if use_kfac else None
        

        MSE = []
        MSE_val = []
        train_acc = []
        val_acc = []
        times = []

        itr = 0
        start_time = time.time()
        stopping_rule = TrainingErrorStoppingRule(patience=5, alpha=0.1, verbose=False)

        for epoch in range(epochs):
            
            model.train()
            for X_batch, y_batch in train_dataloader:
                itr += 1
                
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                logits = model(X_batch)

                # Binary vs Multi-class support
                if logits.shape[-1] == 1:
                    loss = criterion(logits.squeeze(dim=1), y_batch.float())
                else:
                    loss = criterion(logits, y_batch)

                loss.backward()

                # KFAC preconditioner step
                #if precond is not None:
                precond.step()

                optimizer.step()
                stopping_rule(criterion(model(X_batch), y_batch).item())
                # Evaluate on training and test data
                tr= self.evaluate(model, train_dataloader, device)
                val = self.evaluate(model, test_dataloader, device)

                MSE.append(tr[0])
                MSE_val.append(val[0])
                train_acc.append(tr[1])
                val_acc.append(val[1])
                times.append(time.time() - start_time)

                

                if max_iter> 20:
                    break
                if MSE[-1] < 0.01 and train_acc[-1] > 0.95:
                    print("Early stopping triggered.")
                    break

            print(f"KFAC epoch {epoch}, train_loss={MSE[-1]:.4f}, train_acc={train_acc[-1]:.4f}, val_acc={val_acc[-1]:.4f}")
            scheduler.step()
                
            if MSE[-1] < 0.01 and train_acc[-1] > 0.95:
                break

        return model, MSE, MSE_val, train_acc, val_acc, times
        
    def eigen_spectra(self, D):
        SIGMA = []
        VT =[]
        U = []
        for d in D:
            u, s, vt = torch.linalg.svd(d)
            #sigma = (s**2)/(s**4 + 1)
            SIGMA.append(s)
            VT.append(vt)
            U.append(u)
        return U, SIGMA, VT
    
    
    def evaluate(self, model, loader, device):
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
    
        ce_criterion = nn.CrossEntropyLoss()
        mse_criterion = nn.MSELoss()
    
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
    
                output = model(data)                 # [B, C]
    
                # ------------- detect target format automatically -------------
                if target.dim() == 2 and target.size(1) == output.size(1):
                    # Case 1: one-hot encoded → MSE
                    loss = mse_criterion(output, target)
                    target_indices = target.argmax(dim=1)
                elif target.dim() == 1:
                    # Case 2: class indices → CE
                    loss = ce_criterion(output, target)
                    target_indices = target
                else:
                    raise ValueError(
                        f"Target tensor shape {target.shape} is not valid. "
                        f"Expected (B) or (B, num_classes={output.size(1)})."
                    )
                # -----------------------------------------------------------------
    
                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1)
                total_correct += pred.eq(target_indices).sum().item()
                total_samples += data.size(0)
    
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    
    def evaluate_batch(self, model, X_batch, y_batch):
        """
        Evaluate model performance on a single batch.

        :param model: Trained model
        :param X_batch: Input batch tensor
        :param y_batch: Target one-hot encoded labels tensor
        :return: loss
        """
        model.eval()
        criterion = nn.MSELoss()

        with torch.no_grad():
            output = model(X_batch)
            loss = criterion(output, y_batch)

            # Accuracy: convert one-hot to class indices
            pred = output.argmax(dim=1)
            target_indices = y_batch.argmax(dim=1)
            correct = pred.eq(target_indices).sum().item()
            accuracy = correct / X_batch.size(0)

        return loss.item(), accuracy
        

    
