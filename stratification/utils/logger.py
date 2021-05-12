from collections import defaultdict
from datetime import datetime
import logging
import os
import sys

import pandas as pd

from .utils import flatten_dict


class EpochCSVLogger:
    '''Save training process without relying on fixed column names'''

    def __init__(self, fpath, title=None, resume=False):
        self.fpath = fpath
        self.metrics_dict = {}
        if fpath is not None:
            if resume:
                self.metrics_dict = pd.read_csv(fpath, sep='\t').to_dict()
        self.metrics_dict = defaultdict(list, self.metrics_dict)

    def append(self, metrics):
        self.metrics_dict['timestamp'].append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        metrics = flatten_dict(metrics)
        for k, v in metrics.items():
            self.metrics_dict[k].append(f'{v:.6f}')
        pd.DataFrame(self.metrics_dict).to_csv(self.fpath, sep='\t', index=False)

    def close(self):
        pass


class SimpleLogger:
    def __init__(self):
        self.type = 'simple'

    def basic_info(self, message):
        print(message)

    def info(self, message):
        pass

    def warning(self, message):
        print('WARNING:', message)


class FullLogger:
    '''Wrapper class for Python logger'''

    def __init__(self, logger):
        self.type = 'full'
        self.logger = logger

    def basic_info(self, message):
        self.logger.info(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)


def init_epoch_logger(save_dir):
    epoch_log_path = os.path.join(save_dir, 'epochs.tsv')
    epoch_logger = EpochCSVLogger(epoch_log_path)
    logging.info(f'Logging epoch output to {epoch_log_path}.')
    return epoch_logger


def init_logger(name, save_dir, log_format='full'):
    if log_format == 'full':
        log_path = os.path.join(save_dir, 'experiment.log')
        file_handler = logging.FileHandler(filename=log_path, mode='a')
        file_handler.setFormatter(logging.Formatter(fmt='%(asctime)s %(message)s'))
        base_logger = logging.getLogger(name)
        base_logger.addHandler(file_handler)
        base_logger.setLevel(logging.INFO)
        logger = FullLogger(base_logger)
        logging.info('')  # seems to be required to initialize logging properly
        logger.info(f'Logging all output to {log_path}')
        return logger
    else:
        return SimpleLogger()
