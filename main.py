from torch.utils.data import SubsetRandomSampler, DataLoader,TensorDataset
from GResNNSGL.utils import args
from GResNNSGL.models import load_model
from GResNNSGL.utils.load_data import load_data
from GResNNSGL.utils.utils import compute_adaptive_weights
from GResNNSGL.utils import *
from GResNNSGL.opt.training_pipeline import train, validate, objective
import optuna
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import logging
import sys, random, os, math, string
import builtins
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import csv
from itertools import product
from torch.optim import Optimizer
from torch.optim import Adam
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# costumise with your values
job_id = os.environ['SLURM_JOB_ID']
job_name = os.environ['SLURM_JOB_NAME']
method = args.method_name
dataset = args.dataset

# change with your path
group_weight_path = '/nesi/project/uoa03056/MultiOmics/ADNI/GWAS/GWAS_union_ad_genes.csv'


def compute_group_importance(best_model, test_loader, criterion, gene_groups, args):
    """ Computes group importance using permutation-based method """
    original_loss = validate(best_model, test_loader, criterion, args)["loss"]
    group_importance = {}

    for group_idx, group in enumerate(gene_groups):
        permuted_X = test_loader.dataset.tensors[0].clone()  # Copy test data
        permuted_X[:, group] = permuted_X[torch.randperm(permuted_X.size(0)), group]  # Shuffle sample order in the group

        permuted_dataset = TensorDataset(permuted_X, test_loader.dataset.tensors[1])
        permuted_loader = DataLoader(permuted_dataset, batch_size=len(permuted_dataset), shuffle=False)

        permuted_loss = validate(best_model, permuted_loader, criterion, args)["loss"]
        group_importance[group_idx] = original_loss - permuted_loss  # smaller < 0 = more important

    return group_importance


def single_run(seed):
    
    train_stats = None  # Initialize variable to prevent referencing before assignment
    test_stats = None
    
    try:
        fix_random_seed(seed)

        # Define the objective function for Optuna
        objective_fn = lambda trial: objective(trial, seed)

        # Create an Optuna study
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        unique_trials = args.trials
        while unique_trials > len(set(str(t.params) for t in study.trials)):
            study.optimize(objective_fn, n_trials=1, n_jobs=-1)

        # Select the best hyperparameters
        best_params = study.best_trial.params
        max_best_epoch = best_epoch = study.best_trial.user_attrs.get("max_best_epoch", None)

        # Data loading
        kwargs = {} if args.device == 'cpu' else {'num_workers': 2, 'pin_memory': True}
        loader_kwargs = {**kwargs}

        # Load data
        fix_random_seed(seed)
        X_tr, X_te, y_tr,y_te, gene_groups = load_data(seed)

        # Create DataLoaders
        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=len(X_tr), shuffle=True, **loader_kwargs)
        test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=len(X_te), shuffle=False, **loader_kwargs)

        # Initialize the best model with optimal parameters
        best_model = load_model(best_params, gene_groups, args.model).to(args.device)

        # Define loss function
        if args.task_type == 'regression':
            criterion = nn.MSELoss()  
        elif args.task_type == 'binary_class':
            criterion = nn.BCEWithLogitsLoss()  
        else:  
            criterion = nn.CrossEntropyLoss() 
            
        optimizer = optim.Adam(best_model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])

        # Early stopping parameters
        max_epoch = max_best_epoch
        best_val_loss = float('inf')
        best_epoch = 0
        patience = 10
        patience_counter = 0
            
        for epoch in range(max_epoch):
            train_stats = train(best_model, train_loader, optimizer, criterion, best_params['lambda_lasso'], best_params['lambda_group'], gene_groups, args,adaptive_weights = None)
               
        # Evaluate the model on the test set
        test_stats = validate(best_model, test_loader, criterion,args)

        print(f'Best hyperparameters for this split: {best_params}')
        print(f'Train MSE: {train_stats["loss"]}')
        print(f'Test MSE: {test_stats["loss"]}')
        
        if args.task_type == 'regression':
            print(f'Train Corr: {train_stats["correlation"]}')
            print(f'Test Corr: {test_stats["correlation"]}')
        else:
            print(f'Test Accuracy: {test_stats["accuracy"]}')
            print(f'Test F1 Score: {test_stats["f1"]}')
            print(f'Test AUC-ROC: {test_stats["auc_roc"]}')
        
        group_importance = compute_group_importance(best_model, test_loader, criterion, gene_groups, args)
     
        with open(f'/nesi/project/uoa03056/MO_MKL_GL/results/results_{method}_{dataset}.csv', mode='a', newline='') as file:
        #with open(f'/nesi/project/uoa03056/MO_MKL_GL/results/WM/results_GeneResModel_GWAS.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                if args.task_type == 'regression':
                    writer.writerow(['Model', 'Run', 'MSE', 'Correlation', 'Best_Epoch', 'dropout1', 'learning_rate', 'weight_decay', 'lambda_group', 'lambda_lasso', 'hidden1', 'hidden2', 'hidden3'])
                else:
                    writer.writerow(['Model', 'Run', 'Loss', 'Accuracy', 'F1', 'AUC-ROC', 'Best_Epoch', 'dropout1', 'learning_rate', 'weight_decay', 'lambda_group', 'lambda_lasso', 'hidden1', 'hidden2', 'hidden3'])
            if args.task_type == 'regression':
                row = [
                    method, seed, test_stats["loss"], test_stats["correlation"], max_epoch, best_params['dropout'], best_params['learning_rate'], best_params['weight_decay'], best_params['lambda_group'], best_params['lambda_lasso'], best_params['hidden1_size'], best_params['hidden2_size'], best_params['hidden3_size']]

            else:
                row = [
                    method, seed, test_stats["loss"],
                    test_stats["accuracy"], test_stats["f1"], test_stats["auc_roc"],
                    max_epoch, best_params['dropout'], best_params['learning_rate'], best_params['weight_decay'], best_params['lambda_group'], best_params['lambda_lasso'], best_params['hidden1_size'], best_params['hidden2_size'], best_params['hidden3_size']
                ]
            writer.writerow(row)
    
        # Save the model
        save_dir = f'/nesi/nobackup/uoa03056/MO_MKL_GL/save_models/{method}_{dataset}'
        os.makedirs(save_dir, exist_ok=True)
        model_save_path = os.path.join(save_dir, f"{method}_{dataset}_best_model_seed_{seed}.pth")
        torch.save(best_model, model_save_path)
        
        #save final selected groups and all weights for each gene groups.
        # Save the selected features and weights to a CSV file
        base_path_feature = f'/nesi/project/uoa03056/MO_MKL_GL/selected_features/' 
        save_path_feature = os.path.join(base_path_feature, f"{method}_{dataset}")
        os.makedirs(save_path_feature, exist_ok=True)
        
        weight_file_path = os.path.join(save_path_feature, f"{method}_{dataset}_weights_seed_{seed}.csv")
        feature_file_path = os.path.join(save_path_feature, f"{method}_{dataset}_select_feature_seed_{seed}.csv")
        best_model.save_weights_and_groups(weight_file_path, feature_file_path)
        
        importance_file_path = os.path.join(save_path_feature, f"{method}_{dataset}_group_importance_seed_{seed}.csv")
        with open(importance_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Group", "Importance"])
            for group, importance in group_importance.items():
                writer.writerow([group, importance])

                

        return test_stats["loss"], test_stats["correlation"] if args.task_type == 'regression' else (test_stats["accuracy"], test_stats["f1"], test_stats["auc_roc"])
    
    except Exception as e:
        print(f"Error during single_run with seed {seed}: {e}")
        return float("inf"), 0.0

def main():
    results = []
    run = 1
    seed = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))
    
    for r in range(1, run + 1):
        try:
            result = single_run(seed=seed)
            results.append(result)
            print(f"Run {r}: {result}")
        except Exception as e:
            print(f"Run {r}: Failed with seed {seed}, Error: {e}")
            continue  # Skip the failed run
  
        
if __name__ == "__main__":
    main()

    

if __name__ == "__main__":
    main()
