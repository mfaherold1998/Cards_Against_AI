import logging as log
from logging.handlers import RotatingFileHandler
from pathlib import Path
from src.utils import ensure_outdir

class ExcludeInfoFilter(log.Filter):
    "Filter and discard all messages that have exactly the INFO level."
    def filter(self, record):
        return record.levelno != log.INFO

def create_logger (log_name: str = None, log_dir_path: Path|str = "./logs") -> log.Logger:
    """
    Configures and returns a logger with handlers for console and rotary file.
    The log will be saved in a subdirectory with the logger's name.
    """

    logger = log.getLogger(log_name or __name__)
    logger.setLevel("DEBUG")

    # If the logger already has handlers, we return it to avoid duplicates.
    if logger.handlers:
        return logger
    
    outdir = ensure_outdir(log_dir_path)
    log_file_path = outdir / f"{log_name}.log"
    
    # Formatters
    formatter_console = log.Formatter("[{levelname}]: {message}", style="{")
    formatter_file = log.Formatter("{asctime} [{levelname}]: {message}", style="{")
    
    # Handlers
    console_handler = log.StreamHandler()
    console_handler.setLevel("DEBUG")
    console_handler.setFormatter(formatter_console)
    logger.addHandler(console_handler)
 
    info_filter = ExcludeInfoFilter()

    rotating_file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    rotating_file_handler.setLevel("WARNING")
    #rotating_file_handler.addFilter(info_filter)
    rotating_file_handler.setFormatter(formatter_file)
    logger.addHandler(rotating_file_handler)
    
    return logger

