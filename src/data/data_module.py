import os
import numpy as np
import pandas as pd
import torch
from typing import Optional, Tuple, List
from torch.utils.data import DataLoader
import logging

from src.data.dataset import split_numerical_categorical, preprocess_data, TabularDataset
from src.utils.config import Config


class VAEDataModule:
    """
    Data module for VAE training.
    
    Handles loading, preprocessing, and creating data loaders for tabular data.
    """
    
    def __init__(self, config: Config, logger: logging.Logger):
        """
        Initialize the data module.
        
        Args:
            config: Configuration object with data parameters
            logger: Logger instance for logging data processing steps
        """
        self.config = config
        self.logger = logger
        
        # Initialize data attributes
        self.num_mat = None
        self.cat_mat = None
        self.mapping = None
        self.categories = None
        self.d_numerical = None
        self.preprocessed_data = None
        self.train_dataset = None
        self.test_dataset = None
        
    def setup(self) -> None:
        """
        Load and preprocess the data.
        
        Sets up the data for training by:
        1. Loading the data from the specified path
        2. Splitting into numerical and categorical features
        3. Preprocessing the data with optional scaling and encoding
        4. Creating PyTorch datasets
        """
        data_path = self.config.data.train_data_path
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at: {data_path}")
        
        self.logger.info(f"Loading data from {data_path}")
        
        # Load data
        df = pd.read_csv(data_path)
        self.logger.info(f"Loaded dataframe with shape: {df.shape}")
        
        # Split numerical and categorical features
        self.num_mat, self.cat_mat, self.mapping = split_numerical_categorical(
            df, cardinality_threshold=self.config.data.cardinality_threshold
        )
        
        # Convert to correct data types
        self.num_mat = self.num_mat.astype(np.float32)
        self.cat_mat = self.cat_mat.astype(np.int64)
        
        # Get categories
        self.categories = self.get_categories(self.cat_mat)
        self.d_numerical = self.num_mat.shape[1]
        
        self.logger.info(f"Numerical features: {self.d_numerical}")
        if self.categories:
            self.logger.info(f"Categorical features: {len(self.categories)}")
            self.logger.info(f"Categories: {self.categories}")
        
        # Preprocess data
        self.preprocessed_data = preprocess_data(
            num_mat=self.num_mat,
            cat_mat=self.cat_mat,
            mapping=self.mapping,
            y=None,
            test_size=self.config.data.test_size,
            random_state=self.config.seed,
            transform=self.config.data.transform,
            scaling_strategy=self.config.data.scaling_strategy,
            cat_encoding=self.config.data.cat_encoding
        )
        
        # Access the train and test datasets
        self.train_dataset = self.preprocessed_data.train_dataset
        self.test_dataset = self.preprocessed_data.test_dataset
        
        self.logger.info(f"Train dataset size: {len(self.train_dataset)}")
        self.logger.info(f"Test dataset size: {len(self.test_dataset)}")
    
    def get_categories(self, x_train_cat: np.ndarray) -> Optional[List[int]]:
        """
        Get the number of categories for each categorical feature.
        
        Args:
            x_train_cat: Categorical features matrix
            
        Returns:
            List of category counts for each categorical feature or None if there are no categorical features
        """
        return (
            None
            if x_train_cat is None or x_train_cat.shape[1] == 0
            else [
                len(set(x_train_cat[:, i]))
                for i in range(x_train_cat.shape[1])
            ]
        )
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and test data loaders.
        
        Returns:
            Tuple of (train_loader, test_loader)
        """
        if self.train_dataset is None or self.test_dataset is None:
            raise ValueError("Data not set up. Call setup() before getting dataloaders.")
        
        # Determine if we should use pin_memory based on device type
        use_pin_memory = (torch.cuda.is_available() or 
                          (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()))
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=0,  # Can be configured if needed
            pin_memory=use_pin_memory
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=0,  # Can be configured if needed
            pin_memory=use_pin_memory
        )
        
        return train_loader, test_loader
    
    def get_model_dims(self) -> Tuple[int, Optional[List[int]]]:
        """
        Get model dimensions for initializing the VAE.
        
        Returns:
            Tuple of (numerical_features_dim, categorical_features_categories)
        """
        return self.d_numerical, self.categories 