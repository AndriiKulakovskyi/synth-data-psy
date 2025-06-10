import torch
import pandas as pd
from typing import Tuple
from torch.utils.data import Dataset


class IntegratedTabularDataset(Dataset):
    """PyTorch Dataset for preprocessed tabular data from the new pipeline."""
    
    def __init__(self, numerical_df: pd.DataFrame, categorical_df: pd.DataFrame):
        if len(numerical_df) != len(categorical_df):
            raise ValueError("Numerical and categorical dataframes must have same length")
        
        # Convert to tensors
        self.X_num = torch.from_numpy(numerical_df.values).float()
        self.X_cat = torch.from_numpy(categorical_df.values).long()
        
        self.num_features = numerical_df.shape[1]
        self.cat_features = categorical_df.shape[1]
        
    def __len__(self) -> int:
        return len(self.X_num)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X_num[idx], self.X_cat[idx]
