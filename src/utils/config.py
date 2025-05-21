import os
import yaml
from dataclasses import dataclass
from typing import Optional, Dict, Any, List


@dataclass
class TrainingConfig:
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    beta_max: float
    beta_min: float
    beta_decay_factor: float
    early_stopping_patience: int
    scheduler_patience: int
    scheduler_factor: float


@dataclass
class ModelConfig:
    num_layers: int
    d_token: int
    n_head: int
    factor: int
    token_bias: bool


@dataclass
class DataConfig:
    train_data_path: str
    test_size: float
    cardinality_threshold: int
    transform: bool
    scaling_strategy: Optional[str]
    cat_encoding: Optional[str]


@dataclass
class PathConfig:
    checkpoint_dir: str
    model_filename: str
    encoder_filename: str
    decoder_filename: str
    embeddings_filename: str


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    paths: PathConfig
    device: str
    seed: int


def load_config(config_path: str) -> Config:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Config object with nested configuration parameters
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create nested config objects
    data_config = DataConfig(
        train_data_path=config_dict['data']['train_data_path'],
        test_size=config_dict['data']['test_size'],
        cardinality_threshold=config_dict['data']['cardinality_threshold'],
        transform=config_dict['data']['transform'],
        scaling_strategy=config_dict['data']['scaling_strategy'],
        cat_encoding=config_dict['data']['cat_encoding']
    )
    
    model_config = ModelConfig(
        num_layers=config_dict['model']['num_layers'],
        d_token=config_dict['model']['d_token'],
        n_head=config_dict['model']['n_head'],
        factor=config_dict['model']['factor'],
        token_bias=config_dict['model']['token_bias']
    )
    
    training_config = TrainingConfig(
        batch_size=config_dict['training']['batch_size'],
        num_epochs=config_dict['training']['num_epochs'],
        learning_rate=float(config_dict['training']['learning_rate']),
        weight_decay=float(config_dict['training']['weight_decay']),
        beta_max=float(config_dict['training']['beta']['max']),
        beta_min=float(config_dict['training']['beta']['min']),
        beta_decay_factor=float(config_dict['training']['beta']['decay_factor']),
        early_stopping_patience=config_dict['training']['early_stopping_patience'],
        scheduler_patience=config_dict['training']['scheduler_patience'],
        scheduler_factor=float(config_dict['training']['scheduler_factor'])
    )
    
    paths_config = PathConfig(
        checkpoint_dir=config_dict['paths']['checkpoint_dir'],
        model_filename=config_dict['paths']['model_filename'],
        encoder_filename=config_dict['paths']['encoder_filename'],
        decoder_filename=config_dict['paths']['decoder_filename'],
        embeddings_filename=config_dict['paths']['embeddings_filename']
    )
    
    # Create main config object
    return Config(
        data=data_config,
        model=model_config,
        training=training_config,
        paths=paths_config,
        device=config_dict['device'],
        seed=config_dict['seed']
    ) 