import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

class CustomFormatter(logging.Formatter):
    """Custom formatter with colored output for different log levels"""
    
    COLORS = {
        'DEBUG': '\033[0;36m',    # Cyan
        'INFO': '\033[0;32m',     # Green
        'WARNING': '\033[0;33m',  # Yellow
        'ERROR': '\033[0;31m',    # Red
        'CRITICAL': '\033[0;35m', # Purple
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record):
        if hasattr(record, 'color'):
            record.msg = f"{self.COLORS.get(record.levelname, '')}{record.msg}{self.COLORS['RESET']}"
        return super().format(record)

def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console_output: bool = True
) -> logging.Logger:
    """
    Sets up a logger with both file and console handlers.
    
    Args:
        name: Name of the logger
        log_dir: Directory to store log files
        log_file: Specific log file name (if None, will generate based on timestamp)
        level: Logging level
        console_output: Whether to output logs to console
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = CustomFormatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Set up file handler if log_dir is provided
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_dir / log_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Set up console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger

def log_hyperparameters(logger: logging.Logger, hparams: Dict[str, Any]) -> None:
    """
    Log hyperparameters in a clean format.
    
    Args:
        logger: Logger instance
        hparams: Dictionary of hyperparameters
    """
    logger.info("=" * 50)
    logger.info("Hyperparameters:")
    logger.info("-" * 50)
    for key, value in sorted(hparams.items()):
        logger.info(f"{key}: {value}")
    logger.info("=" * 50)

def log_metrics(logger: logging.Logger, metrics: Dict[str, float], step: Optional[int] = None, prefix: str = "") -> None:
    """
    Log metrics in a clean format.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of metric names and values
        step: Optional step number (e.g., epoch or iteration)
        prefix: Optional prefix for the metrics (e.g., 'train' or 'val')
    """
    step_str = f"Step {step} | " if step is not None else ""
    prefix = f"{prefix} " if prefix else ""
    
    metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(f"{step_str}{prefix}Metrics: {metrics_str}")

# Example usage:
# logger = setup_logger('train', log_dir='logs')
# logger.info('Starting training...')
# log_hyperparameters(logger, {'lr': 0.001, 'batch_size': 32})
# log_metrics(logger, {'loss': 0.123, 'accuracy': 0.945}, step=1, prefix='train')