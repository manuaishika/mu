# utils/logger.py
import logging
import os
import sys
import time
from datetime import datetime
import json
from pathlib import Path

class Logger:
    """Custom logger for tracking experiments.
    
    This class provides logging utilities for both console output
    and file-based logging, with support for different verbosity levels
    and structured logging of metrics.
    """
    
    def __init__(self, name, log_dir='logs', console_level=logging.INFO, file_level=logging.DEBUG):
        """Initialize the logger.
        
        Args:
            name (str): Logger name, typically the experiment name
            log_dir (str): Directory to save log files
            console_level (int): Logging level for console output
            file_level (int): Logging level for file output
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        # Create log directory if it doesn't exist
        log_dir = Path(log_dir)
        log_dir.mkdir(exist_ok=True, parents=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"{name}_{timestamp}.log"
        
        # Setup console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Setup file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s] %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Metrics logging
        self.metrics_file = log_dir / f"{name}_{timestamp}_metrics.jsonl"
        
        self.info(f"Logger initialized. Log file: {log_file}")
        
    def debug(self, message):
        """Log debug message.
        
        Args:
            message (str): Debug message
        """
        self.logger.debug(message)
        
    def info(self, message):
        """Log info message.
        
        Args:
            message (str): Info message
        """
        self.logger.info(message)
        
    def warning(self, message):
        """Log warning message.
        
        Args:
            message (str): Warning message
        """
        self.logger.warning(message)
        
    def error(self, message):
        """Log error message.
        
        Args:
            message (str): Error message
        """
        self.logger.error(message)
        
    def critical(self, message):
        """Log critical message.
        
        Args:
            message (str): Critical message
        """
        self.logger.critical(message)
        
    def log_metrics(self, metrics, step):
        """Log metrics as JSON.
        
        Args:
            metrics (dict): Dictionary of metrics to log
            step (int): Current step (e.g., epoch, iteration)
        """
        metrics_record = {
            'timestamp': time.time(),
            'step': step,
            **metrics
        }
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics_record) + '\n')

    def log_model_summary(self, model):
        """Log model architecture summary.
        
        Args:
            model (nn.Module): PyTorch model
        """
        self.info(f"Model: {model.__class__.__name__}")
        self.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        self.debug(str(model))
        
    def log_config(self, config):
        """Log configuration parameters.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.info("Configuration:")
        for key, value in config.items():
            self.info(f"  {key}: {value}")