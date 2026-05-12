import logging
import os
from datetime import datetime


def setup_logger(name: str, log_dir: str = "results/logs/run_logs", level=logging.INFO):
    """
    创建文件+控制台双输出的 logger
    日志文件保存在 log_dir/name_时间戳.log
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加 handler
    if not logger.handlers:
        # 文件输出
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        # 控制台输出
        ch = logging.StreamHandler()
        ch.setLevel(level)

        formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger