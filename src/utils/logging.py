import logging as log
from logging.handlers import RotatingFileHandler
from pathlib import Path

def create_logger (log_name: str = None, log_dir_path: str = "./logs") -> log.Logger:
    """
    Configures and returns a logger with handlers for console and rotary file.
    The log will be saved in a subdirectory with the logger's name.
    """

    logger = log.getLogger(log_name or __name__)
    logger.setLevel("DEBUG")

    # If the logger already has handlers, we return it to avoid duplicates.
    if logger.handlers:
        return logger
    
    # Convert and create the logs dir
    outdir = Path(log_dir_path)
    outdir.mkdir(parents=True, exist_ok=True)
    log_file_path = outdir / f"{log_name}.log"
    
    # Formatters
    formatter_console = log.Formatter("[{levelname}]: {message}", style="{")
    formatter_file = log.Formatter("{asctime} [{levelname}]: {message}", style="{")
    
    # Handlers
    console_handler = log.StreamHandler()
    console_handler.setLevel("DEBUG")
    console_handler.setFormatter(formatter_console)
    logger.addHandler(console_handler)

    rotating_file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    
    rotating_file_handler.setLevel("DEBUG")
    #rotating_file_handler.addFilter(info_filter)
    rotating_file_handler.setFormatter(formatter_file)
    logger.addHandler(rotating_file_handler)
    
    return logger

