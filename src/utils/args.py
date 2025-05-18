import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

def parser():
    PARSER = argparse.ArgumentParser(description='Training parameters.')

    # Model choice
    PARSER.add_argument('--model', default='GeneGroupModel_two_layer', type=str,
                        help='Model architecture to use.')
    
    # method name
    PARSER.add_argument('--method_name', default='MO_MKL_GL', type=str)
    
    # task type
    PARSER.add_argument('--task_type', default='binary_class', type=str,
                       choices=['regression', 'binary_class','multi_class'])
    
    # num_classese
    PARSER.add_argument('--num_classes', default=1, type=int)
    
    
    # dataset name
    PARSER.add_argument('--dataset', default='KEGG', type=str)
    
    # unique trials
    PARSER.add_argument('--trials', default=30, type=int)
    
    # Device choice
    PARSER.add_argument('--device', default='cpu', type=str,
                        choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'],
                        help='Device to run the experiment.')

    # Regularization parameters
    PARSER.add_argument('--penaltytype', default='group_lasso', type=str,
                        choices=['group_lasso', 'sparse_group_lasso', 'none','adaptive_sparse_group_lasso','adaptive_group_lasso'],
                        help='Type of regularization: group_lasso,sparse_group_lasso, none, adaptive_sparse_group_lasso, adaptive_group_lasso')
    
    # Experiment name
    PARSER.add_argument('--exp_name', default='test', type=str)
    
    ARGS = PARSER.parse_args()

    if ARGS.device is None:
        ARGS.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return ARGS


args = parser()

if __name__ == "__main__":
    pass
