#!/usr/bin/env python3
import os
import argparse
import warnings
import torch

from src.utils import setup_logger, load_config
from src.data import VAEDataModule
from src.trainer import VAETrainer

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='VAE Training')
    parser.add_argument('--config', type=str, default='config/vae_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data_path', type=str, 
                        help='Override data path from config')
    parser.add_argument('--device', type=str, 
                        help='Override device from config (cpu, cuda:0, mps, etc.)')
    parser.add_argument('--checkpoint_dir', type=str, 
                        help='Override checkpoint directory from config')
    parser.add_argument('--batch_size', type=int, 
                        help='Override batch size from config')
    parser.add_argument('--epochs', type=int, 
                        help='Override number of epochs from config')
    parser.add_argument('--seed', type=int, 
                        help='Override random seed from config')
    
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create logs directory
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Setup logger
    logger = setup_logger('vae_training', logs_dir)
    logger.info('Starting VAE training')
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f'Loaded configuration from {args.config}')
    
    # Override configuration with command line arguments if provided
    if args.data_path:
        config.data.train_data_path = args.data_path
        logger.info(f'Overriding data path: {args.data_path}')
    
    if args.device:
        config.device = args.device
        logger.info(f'Overriding device: {args.device}')
    
    if args.checkpoint_dir:
        config.paths.checkpoint_dir = args.checkpoint_dir
        logger.info(f'Overriding checkpoint directory: {args.checkpoint_dir}')
    
    if args.batch_size:
        config.training.batch_size = args.batch_size
        logger.info(f'Overriding batch size: {args.batch_size}')
    
    if args.epochs:
        config.training.num_epochs = args.epochs
        logger.info(f'Overriding number of epochs: {args.epochs}')
    
    if args.seed:
        config.seed = args.seed
        logger.info(f'Overriding random seed: {args.seed}')
    
    # Initialize data module
    data_module = VAEDataModule(config, logger)
    data_module.setup()
    train_loader, test_loader = data_module.get_dataloaders()
    d_numerical, categories = data_module.get_model_dims()
    
    # Initialize trainer
    trainer = VAETrainer(config, logger)
    trainer.setup_model(d_numerical, categories)
    
    # Train model
    history = trainer.train(train_loader, test_loader)
    
    logger.info('Training completed')
    

if __name__ == '__main__':
    main() 