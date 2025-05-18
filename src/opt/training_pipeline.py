import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import os
import pandas as pd
from src.utils import args
from src.utils import *
from src.utils.utils import compute_adaptive_weights
from src.models import load_model,sparse_group_lasso_loss
from torch.optim import Optimizer
from torch.optim import Adam
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def train(model, dataloader, optimizer, criterion, l1_lambda, l2_lambda, gene_groups, args, adaptive_weights=None):
    model.train()
    total_loss = 0
    y_pred = []
    y_label = []

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        # Base loss
        base_loss = criterion(outputs, targets)

        # Penalty loss
        penalty = sparse_group_lasso_loss(
            model, 
            l1_lambda, 
            l2_lambda, 
            gene_groups, 
            penalty_type=args.penaltytype, 
            adaptive_weights=adaptive_weights
        )
        
        # Total loss
        loss = base_loss + penalty
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # Collect predictions and labels
        y_pred.extend(outputs.detach().cpu().numpy())
        y_label.extend(targets.detach().cpu().numpy())
  
    if args.task_type == 'binary_class':
        y_pred_binary = [1.0 if p > 0.5 else 0.0 for p in y_pred]
        accuracy = accuracy_score(y_label, y_pred_binary)
        f1 = f1_score(y_label, y_pred_binary)
        auc_roc = roc_auc_score(y_label, y_pred)
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy,
            'f1': f1,
            'auc_roc': auc_roc
        }
    
    elif args.task_type == 'multi_class':

        y_pred_multi = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_label, y_pred_multi)
        weighted_f1 = f1_score(y_label, y_pred_multi, average='weighted')
        auc_roc = roc_auc_score(y_label, y_pred, multi_class='ovr', average='weighted')
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy,
            'weighted_f1': weighted_f1,
            'auc_roc': auc_roc
        }
    
    elif args.task_type == 'regression':
        # Regression metrics
        mse = mean_squared_error(y_label, y_pred)
        correlation, _ = pearsonr(y_label, y_pred)
        
        return {
            'loss': total_loss / len(dataloader),
            'mse': mse,
            'correlation': correlation
        }
    
    
def validate(model, dataloader, criterion, args):
    model.eval()
    total_loss = 0
    y_pred = []
    y_label = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)

            outputs = outputs.view(-1)
            targets = targets.view(-1)

            loss = criterion(outputs, targets)
            total_loss += loss.item()

            y_pred.extend(outputs.detach().cpu().numpy())
            y_label.extend(targets.detach().cpu().numpy())


    if args.task_type == 'binary_class':
        
        y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred] 
        accuracy = accuracy_score(y_label, y_pred_binary)
        f1 = f1_score(y_label, y_pred_binary)
        auc_roc = roc_auc_score(y_label, y_pred) 
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy,
            'f1': f1,
            'auc_roc': auc_roc
        }
    
    elif args.task_type == 'multi_class':
    
        y_pred_multi = np.argmax(y_pred, axis=1)  
        accuracy = accuracy_score(y_label, y_pred_multi)
        weighted_f1 = f1_score(y_label, y_pred_multi, average='weighted')
        auc_roc = roc_auc_score(y_label, y_pred, multi_class='ovr', average='weighted')
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy,
            'f1': weighted_f1,
            'auc_roc': auc_roc
        }
    
    elif args.task_type == 'regression':
       
        mse = mean_squared_error(y_label, y_pred)
        correlation, _ = pearsonr(y_label, y_pred)
        
        return {
            'loss': total_loss / len(dataloader),
            'mse': mse,
            'correlation': correlation
        }
    
    
def objective(trial, seed):
    fix_random_seed(seed=seed)

    if args.penaltytype in ['sparse_group_lasso', 'adaptive_sparse_group_lasso']:
        params = {
            'hidden1_size': trial.suggest_categorical('hidden1_size', [100]),
            'hidden2_size': trial.suggest_categorical('hidden2_size', [100]),
            'hidden3_size': trial.suggest_categorical('hidden3_size', [0]),
            'learning_rate': trial.suggest_float('learning_rate', 0.008,0.5),
            'lambda_lasso': trial.suggest_float('lambda_lasso', 1e-7, 1, log=True),  
            'lambda_group': trial.suggest_float('lambda_group', 1e-8, 1, log=True), 
            'weight_decay': trial.suggest_float('weight_decay',1e-8, 0.01, log=True), 
            'dropout': trial.suggest_float('dropout', 0,0.5)
        }
        
    elif args.penaltytype in ['group_lasso', 'adaptive_group_lasso']:
        params = {
            'hidden1_size': trial.suggest_categorical('hidden1_size', [100]),
            'hidden2_size': trial.suggest_categorical('hidden2_size', [100]),
            'hidden3_size': trial.suggest_categorical('hidden3_size', [0]),
            'learning_rate': trial.suggest_categorical('learning_rate', [0.005]),
            'weight_decay': trial.suggest_categorical('weight_decay', [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5]),
            'lambda_lasso': trial.suggest_categorical('lambda_lasso', [0]),
            'lambda_group': trial.suggest_categorical('lambda_group', [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]),
            'dropout': trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4])
        }

    # Check for duplicate trials
    for t in trial.study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        if t.params == trial.params:
            raise optuna.exceptions.TrialPruned("Duplicate parameter set")

    # Load data
    X_tr1, X_te, y_tr1, y_te, gene_groups = load_data(seed)

    # Initialize 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    fold_val_losses = []
    fold_metrics = []
    best_epochs = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_tr1)):
        print(f"Fold {fold + 1}")

        # Split data
        X_tr, X_val = X_tr1[train_idx], X_tr1[val_idx]
        y_tr, y_val = y_tr1[train_idx], y_tr1[val_idx]
        
        # Create DataLoaders
        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=len(train_idx), shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=len(val_idx),shuffle=True)
        
        # Load model
        model = load_model(params, gene_groups, args.model)
        model = model.to(args.device)
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        
        if args.task_type == 'regression':
            criterion = nn.MSELoss()  
        elif args.task_type == 'binary_class':
            criterion = nn.BCEWithLogitsLoss()  
        else:  
            criterion = nn.CrossEntropyLoss()  

        # Training loop with early stopping
        epochs = 1000
        patience = 10
        best_val_loss = float("inf")
        patience_counter = 0
        best_epoch = 0

        for epoch in range(epochs):
            train_results = train(
                model, train_loader, optimizer, criterion,
                params['lambda_lasso'], params['lambda_group'], gene_groups, args
            )

            # Validate the model
            val_results = validate(model, val_loader, criterion, args)
            val_loss = val_results['loss']
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_epoch = epoch
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                print(f"best epoch is {best_epoch}")
                break
          
        fold_val_losses.append(best_val_loss)
        best_epochs.append(best_epoch)
        
    # Compute the average validation loss and correlation across folds
    avg_val_loss = sum(fold_val_losses) / len(fold_val_losses)
    max_best_epoch = max(best_epochs)
    trial.set_user_attr("max_best_epoch", max_best_epoch)

    print(f"Best average validation loss: {avg_val_loss}")
    
    return avg_val_loss

if __name__ == "__main__":
    pass
