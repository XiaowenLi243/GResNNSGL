import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random

# ----- Random Seed Control -----

def fix_random_seed(seed=None):
    torch.manual_seed(seed) # Fixes CPU seed
    torch.cuda.manual_seed(seed) # Fixes seed on the current GPU
    np.random.seed(seed)
    random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For GPU reproducibility (if using CUDA)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
def store_group_weights(model, gene_groups, l1_lambda):
    group_weights = {}
    selected_features = {}
    selected_groups = []

    idx = 0  
    
    for name, param in model.named_parameters():
        if 'weight' in name:  
            for group_idx, group_size in enumerate(gene_groups):
                group_weight = param[idx:idx+group_size]
                group_weights[group_idx] = group_weight

                # Identify selected features (non-zero weights)
                selected_features[group_idx] = (group_weight != 0).sum().item()

                if selected_features[group_idx] > 0:
                    selected_groups.append(group_idx)

                # Update idx for the next group
                idx += group_size

    return group_weights, selected_features, selected_groups


def get_selected_genes(model, threshold=1e-5):
    """
    Get the genes that were selected by the model (i.e., those with non-zero weights).
    
    Args:
    - model: The trained GeneGroupModel instance.
    - threshold: Weight magnitude below which we consider the feature as not selected.
    
    Returns:
    - selected_genes: List of indices of selected genes.
    """
    selected_genes = []
    for i, layer in enumerate(model.gene_layers):
        # Get the weights for the current gene group
        weights = layer.weight.squeeze()
        
        # If the weight is above the threshold, consider this gene as selected
        if (weights.abs() > threshold).any():
            selected_genes.append(i)
    
    return selected_genes



def get_selected_features(model, gene_idx, threshold=1e-5):
    """
    Get the features selected within a specific gene group (layer).
    
    Args:
    - model: The trained GeneGroupModel instance.
    - gene_idx: Index of the gene group.
    - threshold: Weight magnitude below which we consider the feature as not selected.
    
    Returns:
    - selected_features: List of indices of selected features within the gene group.
    """
    layer = model.gene_layers[gene_idx]
    weights = layer.weight.squeeze()  # Get the weights for this gene group
    
    # Find the features with non-zero weights
    selected_features = (weights.abs() > threshold).nonzero(as_tuple=True)[0]
    
    return selected_features




def plot_weights(model, gene_idx):
    """
    Plot the weights for a specific gene group.
    
    Args:
    - model: The trained GeneGroupModel instance.
    - gene_idx: Index of the gene group.
    """
    layer = model.gene_layers[gene_idx]
    weights = layer.weight.squeeze().detach().numpy()  # Get weights as numpy array
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(weights)), weights)
    plt.xlabel("Feature Index")
    plt.ylabel("Weight Value")
    plt.title(f"Weights for Gene Group {gene_idx}")
    plt.show()

    
    
def compute_adaptive_weights(gwas_weights, gene_groups):
    """
    Compute adaptive weights for adaptive sparse group lasso and adaptive group lasso.

    Args:
    - gwas_weight (list or numpy array): GWAS scores for each gene group.
    - gene_groups (list): List of sizes of each gene group, where each entry represents 
                          the number of features in the corresponding group.
    - epsilon (float): Small constant to prevent division by zero.

    Returns:
    - adaptive_weights (dict): A dictionary with:
        - 'lasso_weights': Uniform weights for individual features (1.0 for all features).
        - 'group_weights': Adaptive weights for groups, inversely proportional to GWAS scores.
    """
    # Validate inputs
    if len(gwas_weights) != len(gene_groups):
        raise ValueError("The length of gwas_weight must match the length of gene_groups.")
    
    # Compute group weights
    group_weights = gwas_weights
    
    # Compute lasso weights for all features (uniform weights set to 1.0)
    lasso_weights = [1.0] * sum(gene_groups)
    
    # Combine results into a dictionary
    adaptive_weights = {
        'lasso_weights': lasso_weights,
        'group_weights': group_weights
    }
    
    return adaptive_weights
