import logging
import os
from datetime import datetime
from pathlib import Path

def setup_logger(name, dataset_code, appliance_name, log_base_dir="logs"):
    """
    Setup logger with separate directories for each dataset and appliance combination
    
    Args:
        name: Logger name
        dataset_code: Dataset code (redd_lf, uk_dale, refit)
        appliance_name: Appliance name
        log_base_dir: Base log directory
        
    Returns:
        Configured logger
    """
    # Convert relative path to absolute path
    if not os.path.isabs(log_base_dir):
        current_file = os.path.abspath(__file__)  # Absolute path of logger.py
        current_dir = os.path.dirname(current_file)  # Directory containing logger.py
        project_root = os.path.dirname(current_dir)  # Project root (assuming logger.py is in utils subdirectory)
        log_base_dir = os.path.join(project_root, log_base_dir)
    
    # Print log base directory for confirmation
    print(f"Log base directory: {log_base_dir}")
    
    # Create dataset and appliance specific log directory
    log_dir = Path(log_base_dir) / dataset_code / appliance_name
    os.makedirs(log_dir, exist_ok=True)
    
    # Create unique log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{name}_{timestamp}.log"
    
    # Get logger
    logger = logging.getLogger(f"{name}_{dataset_code}_{appliance_name}")
    logger.setLevel(logging.INFO)
    
    # Prevent log propagation to root logger
    logger.propagate = False
    
    # If logger already has handlers, return it
    if logger.handlers:
        return logger
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Log initial info
    logger.info(f"Logger setup complete, dataset: {dataset_code}, appliance: {appliance_name}")
    logger.info(f"Log file path: {log_file}")
    
    return logger