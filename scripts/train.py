#!/usr/bin/env python3
"""
CIFAR-10 Training Script

This script runs the training process using configurations from the weights/ directory.
It should be run from the project root directory.

Usage:
    python scripts/train.py --config weights/config_name.json --experiment experiment_name [--fold 0]
"""

import sys
import json
import logging
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.train import run_training
from src.logging import setup_logger, log_hyperparameters


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train CIFAR-10 model with K-fold cross validation"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration JSON file (e.g., weights/basic_cnn_config.json)'
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        help='Name of the experiment for logging and model saving'
    )
    
    parser.add_argument(
        '--fold',
        type=int,
        default=0,
        help='K-fold cross validation fold to use (default: 0)'
    )
    
    parser.add_argument(
        '--validate-config',
        action='store_true',
        help='Only validate the configuration file without training'
    )
    
    return parser.parse_args()


def validate_config_file(config_path: str) -> bool:
    """Validate that the configuration file exists and has required fields."""
    logger = logging.getLogger(__name__)
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return False
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Check required sections
        required_sections = ['model', 'training']
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing required section '{section}' in config file")
                return False
        
        # Check required fields in training section
        training_required = ['batch_size', 'num_epochs']
        training_config = config.get('training', {})
        for field in training_required:
            if field not in training_config:
                logger.error(f"Missing required field 'training.{field}' in config file")
                return False
        
        logger.info(f"Configuration file is valid: {config_path}")
        logger.info(f"Model type: {config.get('model', {}).get('type', 'Unknown')}")
        logger.info(f"Epochs: {training_config.get('num_epochs')}")
        logger.info(f"Batch size: {training_config.get('batch_size')}")
        logger.info(f"K-folds: {training_config.get('k_folds', 5)}")
        
        return True
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        return False
    except Exception as e:
        logger.error(f"Error reading configuration file: {e}")
        return False


def main():
    """Main entry point for training script."""
    args = parse_args()
    
    # Set up logging
    log_dir = project_root / 'logs' / args.experiment
    logger = setup_logger(
        name=f"train_fold{args.fold}",
        log_dir=str(log_dir),
        console_output=True
    )
    
    logger.info("Starting CIFAR-10 training script")
    logger.info(f"Experiment name: {args.experiment}")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Fold: {args.fold}")
    
    # Validate config
    if not validate_config_file(args.config):
        logger.error("Configuration validation failed")
        sys.exit(1)
    
    # Load config for logging
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Log hyperparameters
    log_hyperparameters(logger, config)
    
    # Run training
    try:
        results = run_training(args.config, args.experiment, args.fold)
        
        # Log results
        logger.info("Training completed successfully")
        logger.info("Final Results:")
        logger.info("-" * 50)
        logger.info(f"Best training accuracy: {results['best_train_acc']:.2f}%")
        logger.info(f"Best validation accuracy: {results['best_val_acc']:.2f}%")
        logger.info(f"Best epoch: {results['best_epoch']}")
        logger.info(f"Total training time: {results['total_time']:.2f} seconds")
        logger.info(f"Model saved to: {results['model_path']}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()