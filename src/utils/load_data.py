import numpy as np
from torch.utils.data import Dataset,TensorDataset,DataLoader
import pandas as pd
from .args import args
from sklearn.model_selection import train_test_split
from src.utils import *
import os
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def load_data(seed, dataset=args.dataset):
    # Get the job array index from SLURM
    top =50
    job_name = os.environ['SLURM_JOB_NAME']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Running job array index: {seed}")
    # Set random seed for this job
    fix_random_seed(seed)

    folder = dataset 
    # read in
    comcat_omics = pd.read_csv(f'/nesi/nobackup/uoa03056/DL_prediction/Models/sim/sim_data/{folder}/combined_omics_{seed}.csv')

    x1_selected = comcat_omics[comcat_omics['omic_index'] == 1].values[:,:-2].T
    x2_selected = comcat_omics[comcat_omics['omic_index'] == 2].values[:,:-2].T
    x3_selected = comcat_omics[comcat_omics['omic_index'] == 3].values[:,:-2].T

    merged_matrix = np.hstack((x1_selected,x2_selected,x3_selected))

    gene_index = comcat_omics['gene_index']

    # concat them and set omics as gene i
    gene_len = np.unique(gene_index).shape[0]

    # the gene_group is a list of group size
    grouped_features = []
    gene_groups = []

    unique_genes = np.unique(gene_index)

    for gene in unique_genes:
        gene_indices = np.where(gene_index == gene)[0] 
        grouped_features.append(merged_matrix[:, gene_indices]) 
        gene_groups.append(len(gene_indices))  

    input_data = np.concatenate(grouped_features, axis=1)
    labels = np.load(f'/nesi/nobackup/uoa03056/DL_prediction/Models/sim/sim_data/{folder}/{job_name}_to_binary_outcome_{seed}.npy')
    #labels = np.load(f'/nesi/nobackup/uoa03056/DL_prediction/Models/sim/sim_data/Scenario2_Y/{folder}/{job_name}_to_contibuous_outcome_{seed}.npy')
    #labels = np.load(f'/nesi/nobackup/uoa03056/DL_prediction/Models/sim/sim_data/Scenario3_Y/{folder}/{job_name}_to_contibuous_outcome_{seed}.npy')
    labels= labels.astype(np.float32)
    
    indices = np.arange(len(labels))

    # train test split
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=seed)
    
    X_tr = input_data[train_indices]
    X_te = input_data[test_indices]
    
    y_tr = labels[train_indices]
    y_te = labels[test_indices]
    
    # scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    #y_tr = scaler_y.fit_transform(y_tr.reshape(-1, 1))
    #y_te = scaler_y.transform(y_te.reshape(-1, 1))
    
    X_tr= scaler_X.fit_transform(X_tr)
    X_te = scaler_X.transform(X_te)
    
    
    # Convert to PyTorch tensors
    X_tr = torch.tensor(X_tr, dtype=torch.float32)
    X_te = torch.tensor(X_te, dtype=torch.float32)
    
    y_tr = torch.tensor(y_tr, dtype=torch.float32)
    y_te = torch.tensor(y_te, dtype=torch.float32)
    

    return X_tr, X_te, y_tr, y_te, gene_groups


if __name__ == "__main__":
    pass
