import torch.nn as nn
import torch
from src.utils import args
import os
from sklearn.decomposition import KernelPCA
import torch.nn.functional as F
import csv
import pickle
import csv
import pandas as pd

'''
def sparse_group_lasso_loss(model, l1_lambda, l2_lambda, gene_groups, penalty_type= args.penaltytype, adaptive_weights=None):
    """
    Compute the sparse group lasso, group lasso, adaptive sparse group lasso, or adaptive group lasso penalty.

    Args:
    - model: The GeneGroupModel instance.
    - l1_lambda: Regularization weight for lasso penalty.
    - l2_lambda: Regularization weight for group lasso penalty.
    - gene_groups: List of sizes of each gene group.
    - penalty_type: Type of penalty ('sparse_group_lasso', 'group_lasso', 'adaptive_sparse_group_lasso', or 'adaptive_group_lasso').
    - adaptive_weights: A dictionary with 'lasso_weights' and/or 'group_weights' (required for adaptive penalties).

    Returns:
    - The total penalty based on the chosen penalty type.
    """
    lasso_penalty = 0.0
    group_lasso_penalty = 0.0

    for i, layer in enumerate(model.gene_layers):
        # Get weights for the current group
        group_weights = layer.weight.squeeze()
        group_size = gene_groups[i]

        if penalty_type == 'sparse_group_lasso':
            # Lasso penalty
            lasso_penalty += group_weights.abs().sum()
            # Group lasso penalty
            group_lasso_penalty += torch.sqrt(torch.tensor(group_size, dtype=torch.float32)) * group_weights.norm(2)

        elif penalty_type == 'group_lasso':
            # Only group lasso penalty
            group_lasso_penalty += torch.sqrt(torch.tensor(group_size, dtype=torch.float32)) * group_weights.norm(2)

        elif penalty_type == 'adaptive_sparse_group_lasso':
            if adaptive_weights is None:
                raise ValueError("adaptive_weights must be provided for adaptive sparse group lasso.")

            # Adaptive weights
            lasso_weights = adaptive_weights['lasso_weights'][i]
            group_weight = adaptive_weights['group_weights'][i]

            # Adaptive lasso penalty
            lasso_penalty += (lasso_weights * group_weights.abs()).sum()
            # Adaptive group lasso penalty
            group_lasso_penalty += group_weight * group_weights.norm(2)

        elif penalty_type == 'adaptive_group_lasso':
            if adaptive_weights is None:
                raise ValueError("adaptive_weights must be provided for adaptive group lasso.")

            # Adaptive group lasso penalty
            group_weight = adaptive_weights['group_weights'][i]
            group_lasso_penalty += group_weight * group_weights.norm(2)

        else:
            raise ValueError("Invalid penalty_type. Choose 'sparse_group_lasso', 'group_lasso', 'adaptive_sparse_group_lasso', or 'adaptive_group_lasso'.")

    # Combine penalties based on penalty type
    if penalty_type in ['sparse_group_lasso', 'adaptive_sparse_group_lasso']:
        return l1_lambda * lasso_penalty + l2_lambda * group_lasso_penalty
    elif penalty_type in ['group_lasso', 'adaptive_group_lasso']:
        return l2_lambda * group_lasso_penalty
'''

def sparse_group_lasso_loss(model, l1_lambda, l2_lambda, gene_groups, penalty_type='adaptive_sparse_group_lasso', adaptive_weights=None):
    
    """
    Compute the sparse group lasso, group lasso, adaptive sparse group lasso, or adaptive group lasso penalty.

    Args:
    - model: The GeneGroupModel instance.
    - l1_lambda: Regularization weight for lasso penalty.
    - l2_lambda: Regularization weight for group lasso penalty.
    - gene_groups: List of sizes of each gene group.
    - penalty_type: Type of penalty ('sparse_group_lasso', 'group_lasso', 'adaptive_sparse_group_lasso', or 'adaptive_group_lasso').
    - adaptive_weights: A dictionary with 'lasso_weights' and/or 'group_weights' (optional; if None, uses attention scores).

    Returns:
    - The total penalty based on the chosen penalty type.
    """
    lasso_penalty = 0.0
    group_lasso_penalty = 0.0

    for i, layer in enumerate(model.gene_layers):
        # Get weights for the current group
        group_weights = layer.weight.squeeze()
        group_size = gene_groups[i]

        if penalty_type == 'sparse_group_lasso':
            # Lasso penalty
            lasso_penalty += group_weights.abs().sum()
            # Group lasso penalty
            group_lasso_penalty += torch.sqrt(torch.tensor(group_size, dtype=torch.float32)) * group_weights.norm(2)

        elif penalty_type == 'group_lasso':
            # Only group lasso penalty
            group_lasso_penalty += torch.sqrt(torch.tensor(group_size, dtype=torch.float32)) * group_weights.norm(2)

        elif penalty_type == 'adaptive_sparse_group_lasso':
            # Use provided adaptive weights or fallback to attention scores
            if adaptive_weights is None:
                group_attention_score = model.group_attention_layer(group_weights)  
                feature_attention_scores = model.feature_attention_layers[i](group_weights) 

                # Ensure the attention scores are normalized
                assert torch.isclose(group_attention_score.sum(), torch.tensor(1.0)), "Group attention scores must sum to 1"
                assert torch.isclose(feature_attention_scores.sum(), torch.tensor(1.0)), "Feature attention scores must sum to 1"

                # Adaptive lasso penalty
                lasso_penalty += (feature_attention_scores * group_weights.abs()).sum()
                # Adaptive group lasso penalty
                group_lasso_penalty += group_attention_score * group_weights.norm(2)
            else:
                # Use provided adaptive weights
                lasso_weights = adaptive_weights['lasso_weights'][i]
                group_weight = adaptive_weights['group_weights'][i]

                # Adaptive lasso penalty
                lasso_penalty += (lasso_weights * group_weights.abs()).sum()
                # Adaptive group lasso penalty
                group_lasso_penalty += group_weight * group_weights.norm(2)

        elif penalty_type == 'adaptive_group_lasso':
            # Use provided adaptive weights or fallback to attention scores
            if adaptive_weights is None:
                group_attention_score = model.group_attention[i]  

                # Ensure the attention scores are normalized
                assert torch.isclose(group_attention_score.sum(), torch.tensor(1.0)), "Group attention scores must sum to 1"

                # Adaptive group lasso penalty
                group_lasso_penalty += group_attention_score * group_weights.norm(2)
            else:
                # Use provided adaptive weights
                group_weight = adaptive_weights['group_weights'][i]

                # Adaptive group lasso penalty
                group_lasso_penalty += group_weight * group_weights.norm(2)

        else:
            raise ValueError("Invalid penalty_type. Choose 'sparse_group_lasso', 'group_lasso', 'adaptive_sparse_group_lasso', or 'adaptive_group_lasso'.")

    # Combine penalties based on penalty type
    if penalty_type in ['sparse_group_lasso', 'adaptive_sparse_group_lasso']:
        return l1_lambda * lasso_penalty + l2_lambda * group_lasso_penalty
    elif penalty_type in ['group_lasso', 'adaptive_group_lasso']:
        return l2_lambda * group_lasso_penalty
    

class GeneGroupModel(nn.Module):
    def __init__(self, params, gene_groups):
        super(GeneGroupModel, self).__init__()
        
        hidden1_size, hidden2_size, hidden3_size, output_size = params['hidden1_size'], params['hidden2_size'], params['hidden3_size'], 1
        
        # Gene-specific linear layers
        self.gene_layers = nn.ModuleList([nn.Linear(group, 1) for group in gene_groups])
        
        # Fully connected layers
        self.hidden1 = nn.Linear(len(gene_groups), hidden1_size)
        self.hidden2 = nn.Linear(hidden1_size, hidden2_size)
        self.hidden3 = nn.Linear(hidden2_size, hidden3_size)
        self.output = nn.Linear(hidden3_size, output_size)
        
        # Batch Normalization layers
        self.batch_norm1 = nn.BatchNorm1d(hidden1_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden2_size)
        self.batch_norm3 = nn.BatchNorm1d(hidden3_size)
        
        # Activation function
        self.activation = nn.ReLU()
        
        # Dropout layers
        self.dropout1 = nn.Dropout(params['dropout'])  # Dropout for the hidden layers

    def forward(self, x):
        gene_outputs = []
        start_idx = 0
        
        for i, layer in enumerate(self.gene_layers):
            # Process each gene group, apply linear layer followed by ReLU activation
            group_features = x[:, start_idx:start_idx + layer.in_features]
            g_i = self.activation(layer(group_features))
            gene_outputs.append(g_i)
            start_idx += layer.in_features
        
        # Concatenate outputs of all gene layers
        gene_layer_output = torch.cat(gene_outputs, dim=1)
        
        # Pass through hidden layers with batch normalization and dropout
        x = self.activation(self.batch_norm1(self.hidden1(gene_layer_output)))
        x = self.dropout1(x) 
        x = self.activation(self.batch_norm2(self.hidden2(x)))
        x = self.dropout1(x)
        x = self.activation(self.batch_norm3(self.hidden3(x)))
        x = self.dropout1(x) 
        
        # Final output layer
        x = self.output(x)
        
        return x
    
class GeneGroupModel_two_layer(nn.Module):
    def __init__(self, params, gene_groups):
        super(GeneGroupModel_two_layer, self).__init__()
        
        hidden1_size, hidden2_size, output_size = params['hidden1_size'], params['hidden2_size'], 1
        
        # Gene-specific linear layers
        self.gene_layers = nn.ModuleList([nn.Linear(group, 1) for group in gene_groups])
        
        # Fully connected layers
        self.hidden1 = nn.Linear(len(gene_groups), hidden1_size)
        self.hidden2 = nn.Linear(hidden1_size, hidden2_size)
        
        # Final output layer
        self.output_layer = nn.Linear(hidden2_size, output_size)
        
        # Residual connection: Shortcut directly from gene layer output to final output
        #self.residual_output_layer = nn.Linear(len(gene_groups), output_size)

        # Batch Normalization layers
        self.batch_norm1 = nn.BatchNorm1d(hidden1_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden2_size)
        
        # Activation function
        self.activation = nn.ReLU()
        
        # Dropout layers
        self.dropout = nn.Dropout(params['dropout'])

        # Initialize weights using Xavier initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.gene_layers:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.hidden1.weight)
        nn.init.zeros_(self.hidden1.bias)

        nn.init.xavier_uniform_(self.hidden2.weight)
        nn.init.zeros_(self.hidden2.bias)

        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

        #nn.init.xavier_uniform_(self.residual_output_layer.weight)
        #nn.init.zeros_(self.residual_output_layer.bias)

    def forward(self, x):
        # Process each gene group
        gene_outputs = []
        start_idx = 0
        
        for i, layer in enumerate(self.gene_layers):
            group_features = x[:, start_idx:start_idx + layer.in_features]
            g_i = self.activation(layer(group_features)) 
            gene_outputs.append(g_i)
            start_idx += layer.in_features
        
        # Concatenate outputs of all gene layers
        gene_layer_output = torch.cat(gene_outputs, dim=1)
        
        # Main path through fully connected layers
        x_main = self.activation(self.batch_norm1(self.hidden1(gene_layer_output)))
        x_main = self.dropout(x_main)
        x_main = self.activation(self.batch_norm2(self.hidden2(x_main)))
        x_main = self.dropout(x_main)
        x_main = self.output_layer(x_main)
        
        # Residual connection: Gene layer output directly contributes to the final output
        #x_residual = self.residual_output_layer(gene_layer_output)
        
        # Combine the main path and residual connection
        #x = x_main + x_residual
        
        return x_main

    def save_weights_and_groups(self, path_weights='gene_layer_weights.csv', path_groups='selected_groups.csv'):
        # Extract weights from the first layer of each gene group
        gene_layer_weights = []
        selected_groups = []
        
        for i, layer in enumerate(self.gene_layers):
            weights = layer.weight.data.cpu().numpy().flatten()  # Flatten to 1D for easier saving
            gene_layer_weights.append(weights)
            
            # Check if any weight in the group is greater than 0, if yes, save the group
            if any(w > 0 for w in weights):
                selected_groups.append({
                    'group_index': i,
                    'selected_features': list(range(len(weights)))  # Store feature indices
                })
        
        # Convert the gene layer weights to a DataFrame and save to CSV
        gene_layer_weights_df = pd.DataFrame(gene_layer_weights)
        gene_layer_weights_df.to_csv(path_weights, index=False)
        
        # Convert the selected groups to a DataFrame and save to CSV
        selected_groups_df = pd.DataFrame(selected_groups)
        selected_groups_df.to_csv(path_groups, index=False)

    

class GeneAttentionModel(nn.Module):
    def __init__(self, params, gene_groups):
        super(GeneAttentionModel, self).__init__()
        
        hidden1_size, hidden2_size, output_size = (
            params['hidden1_size'], params['hidden2_size'], 1
        )
        
        # Gene-specific linear layers
        self.gene_layers = nn.ModuleList([nn.Linear(group, 1) for group in gene_groups])
        
        # Attention layer for gene outputs
        self.attention_weights = nn.Parameter(torch.randn(len(gene_groups)))
        
        # Fully connected layers
        self.hidden1 = nn.Linear(len(gene_groups), hidden1_size)
        self.hidden2 = nn.Linear(hidden1_size, hidden2_size)
        self.output = nn.Linear(hidden2_size, output_size)
        
        # Batch Normalization layers
        self.batch_norm1 = nn.BatchNorm1d(hidden1_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden2_size)

        # Activation function
        self.activation = nn.LeakyReLU(0.05)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(params['dropout'])  # Dropout for the hidden layers

    def forward(self, x):
        gene_outputs = []
        start_idx = 0
        
        for i, layer in enumerate(self.gene_layers):
            # Process each gene group, apply linear layer followed by ReLU activation
            group_features = x[:, start_idx:start_idx + layer.in_features]
            g_i = self.activation(layer(group_features))
            gene_outputs.append(g_i)
            start_idx += layer.in_features
        
        # Concatenate outputs of all gene layers
        gene_layer_output = torch.cat(gene_outputs, dim=1)  # Shape: (batch_size, num_groups)
        
        # Compute attention weights
        attention_scores = F.softmax(self.attention_weights, dim=0)  # Normalize attention weights
        attention_scores = attention_scores.unsqueeze(0).expand_as(gene_layer_output)  # Match batch size
        
        # Apply attention to gene layer outputs
        attended_gene_output = gene_layer_output * attention_scores  
        
        # Pass through hidden layers with batch normalization and dropout
        x = self.activation(self.batch_norm1(self.hidden1(attended_gene_output)))
        x = self.dropout1(x)
        x = self.activation(self.batch_norm2(self.hidden2(x)))
        x = self.dropout1(x)
        
        # Final output layer
        x = self.output(x)
        
        return x


class GeneResModel_two_layer(nn.Module):
    def __init__(self, params, gene_groups, task_type=args.task_type, num_classes = args.num_classes):
        super(GeneResModel_two_layer, self).__init__()
        
        self.task_type = task_type
        
        hidden1_size, hidden2_size, output_size = params['hidden1_size'], params['hidden2_size'], num_classes
        
        # Gene-specific linear layers
        self.gene_layers = nn.ModuleList([nn.Linear(group, 1) for group in gene_groups])
        
        # Fully connected layers
        self.hidden1 = nn.Linear(len(gene_groups), hidden1_size)
        self.hidden2 = nn.Linear(hidden1_size, hidden2_size)
        
        # Final output layer
        self.output_layer = nn.Linear(hidden2_size, output_size)
        
        # Residual connection: Shortcut directly from gene layer output to final output
        self.residual_output_layer = nn.Linear(len(gene_groups), output_size)

        # Batch Normalization layers
        self.batch_norm1 = nn.BatchNorm1d(hidden1_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden2_size)
        
        # Activation function
        self.activation = nn.ReLU()
        
        # Dropout layers
        self.dropout = nn.Dropout(params['dropout'])

        # Initialize weights using Xavier initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.gene_layers:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.hidden1.weight)
        nn.init.zeros_(self.hidden1.bias)

        nn.init.xavier_uniform_(self.hidden2.weight)
        nn.init.zeros_(self.hidden2.bias)

        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

        nn.init.xavier_uniform_(self.residual_output_layer.weight)
        nn.init.zeros_(self.residual_output_layer.bias)

    def forward(self, x):
        # Process each gene group
        gene_outputs = []
        start_idx = 0
        
        for i, layer in enumerate(self.gene_layers):
            group_features = x[:, start_idx:start_idx + layer.in_features]
            g_i = self.activation(layer(group_features)) 
            gene_outputs.append(g_i)
            start_idx += layer.in_features
        
        # Concatenate outputs of all gene layers
        gene_layer_output = torch.cat(gene_outputs, dim=1)
        
        # Main path through fully connected layers
        x_main = self.activation(self.batch_norm1(self.hidden1(gene_layer_output)))
        x_main = self.dropout(x_main)
        x_main = self.activation(self.batch_norm2(self.hidden2(x_main)))
        x_main = self.dropout(x_main)
        x_main = self.output_layer(x_main)
        
        # Residual connection: Gene layer output directly contributes to the final output
        x_residual = self.residual_output_layer(gene_layer_output)
        
        # Combine the main path and residual connection
        x_combined = x_main + x_residual
        
        # Apply activation based on task type
        if self.task_type == 'binary_class':
            x = torch.sigmoid(x_combined)  # Sigmoid for binary classification
        elif self.task_type == 'regression':
            x = x_combined  # No activation for regression
        elif self.task_type == 'multi_class':
            x = torch.softmax(x_combined, dim=1)  # Softmax for multi-class classification
        
        return x

    def save_weights_and_groups(self, path_weights='gene_layer_weights.csv', path_groups='selected_groups.csv'):
        # Extract weights from the first layer of each gene group
        gene_layer_weights = []
        selected_groups = []
        
        for i, layer in enumerate(self.gene_layers):
            weights = layer.weight.data.cpu().numpy().flatten()  # Flatten to 1D for easier saving
            gene_layer_weights.append(weights)
            
            # Check if any weight in the group is greater than 0, if yes, save the group
            if any(w > 0 for w in weights):
                selected_groups.append({
                    'group_index': i,
                    'selected_features': list(range(len(weights)))  # Store feature indices
                })
        
        # Convert the gene layer weights to a DataFrame and save to CSV
        gene_layer_weights_df = pd.DataFrame(gene_layer_weights)
        gene_layer_weights_df.to_csv(path_weights, index=False)
        
        # Convert the selected groups to a DataFrame and save to CSV
        selected_groups_df = pd.DataFrame(selected_groups)
        selected_groups_df.to_csv(path_groups, index=False)


class GeneResModelTwoLayer(nn.Module):
    def __init__(self, params, gene_groups):
        super(GeneResModelTwoLayer, self).__init__()
        
        hidden1_size, hidden2_size, output_size = params['hidden1_size'], params['hidden2_size'], 1
        
        # Gene-specific linear layers
        self.gene_layers = nn.ModuleList([nn.Linear(group, 1) for group in gene_groups])
                
        # Calculate split index dynamically
        self.num_groups = len(gene_groups)
        self.split_idx = self.num_groups - 3  # Last 3 groups are separated
        
        # Separate fully connected layers for the first groups
        self.hidden1_group1 = nn.Linear(self.split_idx, hidden1_size)
        self.hidden2_group1 = nn.Linear(hidden1_size, hidden2_size)
        
        # Separate fully connected layers for the last 3 groups
        self.hidden1_group2 = nn.Linear(3, hidden1_size)
        self.hidden2_group2 = nn.Linear(hidden1_size, hidden2_size)
        
        
        # Final output layer combining both groups
        self.output_layer = nn.Linear(2 * hidden2_size, output_size)
        
        # Residual connection: Direct shortcut from gene layer output to final output
        self.residual_output_layer = nn.Linear(self.num_groups, output_size)
        
        # Batch Normalization layers
        self.batch_norm1_group1 = nn.BatchNorm1d(hidden1_size)
        self.batch_norm2_group1 = nn.BatchNorm1d(hidden2_size)
        self.batch_norm1_group2 = nn.BatchNorm1d(hidden1_size)
        self.batch_norm2_group2 = nn.BatchNorm1d(hidden2_size)
        
        # Activation function
        self.activation = nn.ReLU()
        
        # Dropout layers
        self.dropout = nn.Dropout(params['dropout'])

        # Initialize weights using Xavier initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.gene_layers:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        
        nn.init.xavier_uniform_(self.hidden1_group1.weight)
        nn.init.zeros_(self.hidden1_group1.bias)
        nn.init.xavier_uniform_(self.hidden2_group1.weight)
        nn.init.zeros_(self.hidden2_group1.bias)
        
        nn.init.xavier_uniform_(self.hidden1_group2.weight)
        nn.init.zeros_(self.hidden1_group2.bias)
        nn.init.xavier_uniform_(self.hidden2_group2.weight)
        nn.init.zeros_(self.hidden2_group2.bias)
        
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        
        nn.init.xavier_uniform_(self.residual_output_layer.weight)
        nn.init.zeros_(self.residual_output_layer.bias)

    def forward(self, x):
        # Process each gene group
        gene_outputs = []
        start_idx = 0
        
        for i, layer in enumerate(self.gene_layers):
            group_features = x[:, start_idx:start_idx + layer.in_features]
            g_i = self.activation(layer(group_features))
            gene_outputs.append(g_i)
            start_idx += layer.in_features
        
        # Concatenate outputs of all gene layers
        gene_layer_output = torch.cat(gene_outputs, dim=1)
        
        # Split the gene layer output dynamically
        group1_output = gene_layer_output[:, :self.split_idx]
        group2_output = gene_layer_output[:, self.split_idx:]
        
        # Process group 1 through its hidden layers
        x_group1 = self.activation(self.batch_norm1_group1(self.hidden1_group1(group1_output)))
        x_group1 = self.dropout(x_group1)
        x_group1 = self.activation(self.batch_norm2_group1(self.hidden2_group1(x_group1)))
        x_group1 = self.dropout(x_group1)
        
        # Process group 2 through its hidden layers
        x_group2 = self.activation(self.batch_norm1_group2(self.hidden1_group2(group2_output)))
        x_group2 = self.dropout(x_group2)
        x_group2 = self.activation(self.batch_norm2_group2(self.hidden2_group2(x_group2)))
        x_group2 = self.dropout(x_group2)
        
        # Combine the outputs from both groups
        combined_output = torch.cat([x_group1, x_group2], dim=1)
        
        # Main path output
        x_main = self.output_layer(combined_output)
        
        # Residual connection: Gene layer output directly contributes to the final output
        x_residual = self.residual_output_layer(gene_layer_output)
        
        # Combine the main path and residual connection
        x = x_main + x_residual
        
        return x
    
    def save_weights_and_groups(self, path_weights='gene_layer_weights.csv', path_groups='selected_groups.csv'):
        # Extract weights from the first layer of each gene group
        gene_layer_weights = []
        selected_groups = []
        
        for i, layer in enumerate(self.gene_layers):
            weights = layer.weight.data.cpu().numpy().flatten()  # Flatten to 1D for easier saving
            gene_layer_weights.append(weights)
            
            # Check if any weight in the group is greater than 0, if yes, save the group
            if any(w > 0 for w in weights):
                selected_groups.append({
                    'group_index': i,
                    'selected_features': list(range(len(weights)))  # Store feature indices
                })
        
        # Convert the gene layer weights to a DataFrame and save to CSV
        gene_layer_weights_df = pd.DataFrame(gene_layer_weights)
        gene_layer_weights_df.to_csv(path_weights, index=False)
        
        # Convert the selected groups to a DataFrame and save to CSV
        selected_groups_df = pd.DataFrame(selected_groups)
        selected_groups_df.to_csv(path_groups, index=False)
    
    

class GeneResAttenSGL_2l(nn.Module):
    def __init__(self, params, gene_groups):
        super(GeneResAttenSGL_2l, self).__init__()

        hidden1_size, hidden2_size, output_size = (
            params['hidden1_size'], params['hidden2_size'], 1
        )

        # Gene-specific linear layers
        self.gene_layers = nn.ModuleList([nn.Linear(group, 1) for group in gene_groups])

        # Group-level attention
        self.group_attention_layer = nn.Linear(len(gene_groups), len(gene_groups))

        # Feature-level attention for each group
        self.feature_attention_layers = nn.ModuleList(
            [nn.Linear(group, group) for group in gene_groups]
        )

        # Fully connected layers
        self.hidden1 = nn.Linear(len(gene_groups), hidden1_size)
        self.hidden2 = nn.Linear(hidden1_size, hidden2_size)

        # Final output layer
        self.output_layer = nn.Linear(hidden2_size, output_size)

        # Shortcut connection from gene layer output to final output
        self.shortcut_output = nn.Linear(len(gene_groups), output_size)

        # Linear transformation to combine paths
        self.combined_layer = nn.Linear(output_size * 2, output_size)

        self.batch_norm1 = nn.BatchNorm1d(hidden1_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden2_size)

        # Activation function
        self.activation = nn.ReLU()

        # Dropout layers
        self.dropout1 = nn.Dropout(params['dropout'])

    def forward(self, x):  
        gene_outputs = []
        start_idx = 0
        feature_attention_scores_list = []
        selected_features = []
        group_weights = []
            
        for i, layer in enumerate(self.gene_layers):
            group_features = x[:, start_idx:start_idx + layer.in_features]

            # Feature-level attention
            feature_attention_logits = self.feature_attention_layers[i](group_features)
            feature_attention_scores = F.softmax(feature_attention_logits, dim=1)
            feature_attention_scores_list.append(feature_attention_scores)

            # Store selected features and weights
            selected_features.append(group_features)
            group_weights.append(layer.weight.squeeze())

            g_i = self.activation(layer(group_features))
            gene_outputs.append(g_i)
            start_idx += layer.in_features

        gene_layer_output = torch.cat(gene_outputs, dim=1)
        group_attention_logits = self.group_attention_layer(gene_layer_output)
        group_attention_scores = F.softmax(group_attention_logits, dim=1)

        attended_gene_outputs = [
            g_i * group_attention_scores[:, i].unsqueeze(1) for i, g_i in enumerate(gene_outputs)
        ]
        gene_layer_output = torch.cat(attended_gene_outputs, dim=1)

        x_main = self.activation(self.batch_norm1(self.hidden1(gene_layer_output)))
        x_main = self.dropout1(x_main)
        x_main = self.activation(self.batch_norm2(self.hidden2(x_main)))
        x_main = self.dropout1(x_main)
        x_main = self.output_layer(x_main)

        x_shortcut = self.shortcut_output(gene_layer_output)
        combined = torch.cat((x_main, x_shortcut), dim=1)
        x = self.combined_layer(combined)

        # Save selected features, group weights, and attention scores
        self.selected_features = selected_features
        self.group_weights = group_weights
        self.feature_attention_scores = feature_attention_scores_list
        self.group_attention_scores = group_attention_scores

        return x

    def save_selected_features_and_weights(self, file_path='selected_features_and_weights.pkl'):
        """
        Save the selected features, group weights, and corresponding attention scores to a file.
        """
        data = {
            'selected_features': self.selected_features,
            'group_weights': self.group_weights,
            'feature_attention_scores': self.feature_attention_scores,
            'group_attention_scores': self.group_attention_scores
        }

        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    

    
def load_model(params, gene_groups, model_name=args.model):
    # Model selected
    if model_name == 'GeneGroupModel':
        model= GeneGroupModel(params, gene_groups)
    elif model_name == 'GeneResModel_two_layer':
        model=GeneResModel_two_layer(params, gene_groups, task_type=args.task_type, num_classes = args.num_classes)
    elif model_name == 'GeneResModelTwoLayer':
        model=GeneResModelTwoLayer(params, gene_groups)
    elif model_name == 'GeneAttentionModel':
        model=GeneAttentionModel(params, gene_groups)
    elif model_name == 'GeneGroupModel_two_layer':
        model=GeneGroupModel_two_layer(params, gene_groups)
    elif model_name == 'GeneResAttenSGL_2l':
        model=GeneResAttenSGL_2l(params, gene_groups) 
        
        
    model = model.to(args.device)
    
    return model

if __name__ == "__main__":
    pass

