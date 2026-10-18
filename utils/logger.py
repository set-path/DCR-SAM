import logging
import os
import sys

def setup_logger(name, save_dir, distributed_rank, filename="log.txt"):
    """
    Configure and return a standard logger that only prints on rank 0 in distributed runs.
    """
    logger = logging.getLogger(name)
    
    logger.propagate = False
    
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers = []

    if distributed_rank > 0:
        return logger

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(save_dir, filename), encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger