# Implementation to setup logger

import logging
import logging.config
from pathlib import Path
from utils import read_json

def setup_logging(saveDirectory, loggerConfig="logger/logger_config.json", defaultLevel=logging.INFO):
    """
    Function to setup loggining.

    Parameters
    ----------
    saveDirectory   : str
                      Name of the directory
    loggerConfig    : str
                      Path to the logger config file
    defaultLevel    : Logger
                      Logs a message with level INFO on the root logger

    Returns
    -------
    None
    """
    loggerConfig = Path(loggerConfig)
    if loggerConfig.is_file():
        config = read_json(loggerConfig)
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(saveDirectory / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("[WARNING] Logger configuration is not found in the path {}".format(loggerConfig))
        logging.basicConfig(level=defaultLevel)
