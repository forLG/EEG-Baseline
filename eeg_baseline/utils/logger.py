import os
import logging
import time
from addict import Dict

def setup_logger(config: Dict) -> logging.Logger:
    """
    Set up the logger based on the configuration.

    Args:
        config (Dict): Configuration dictionary containing logging settings.

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_filename = f"{config.training.logging.project_name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(config.training.logging.dir, log_filename)

    logger = logging.getLogger(config.training.logging.project_name)
    logger.setLevel(logging.INFO)

    # 防止重复添加 handler (在某些交互式环境中可能发生)
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建文件 handler，写入日志文件
    file_handler = logging.FileHandler(log_filepath, mode='a') # 'a' for append
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 创建控制台 handler，打印到屏幕
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(message)s') # 控制台只输出简洁信息
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger