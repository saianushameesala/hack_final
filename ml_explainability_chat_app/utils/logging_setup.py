"""
Centralized logging configuration for the ML Explainability Chat App.
"""
import os
import logging

def setup_logging(log_level="INFO", log_file=None):
    """
    Setup logging configuration for the application
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file, if None, logs only to console
    """
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Create logs directory if it doesn't exist
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure root logger
    handlers = []
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    handlers.append(logging.StreamHandler())
    
    # Set format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=handlers
    )
    
    # Configure module-specific loggers
    modules = [
        "app", "vector_store", "llm_connector", "llm_explainer", 
        "parser", "executor", "explainers", "debug"
    ]
    
    for module in modules:
        logger = logging.getLogger(module)
        logger.setLevel(numeric_level)
    
    logging.info(f"Logging initialized at level {log_level}")
    
    return logging.getLogger("app")

def get_logger(name):
    """
    Get a logger for a specific module
    
    Args:
        name: Name of the module
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
