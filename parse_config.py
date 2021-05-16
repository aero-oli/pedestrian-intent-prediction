# Implementation to parse configuration file

import os
import logging
from pathlib import Path
from operator import getitem
from datetime import datetime
from logger import setup_logging
from functools import reduce, partial
from utils import read_json, write_json

class ConfigParser:
    """
    """
    def __init__(self, configuration, resume=None, modification=None, runId=None):
        """
        """
        # Load Configuration File and Apply Modificaitons
        self._configuration = _update_config(configuration, modification)
        self.resume = resume

        # Set Save directory where trained model and the log would be saved
        saveDirectory = Path(self._configuration["trainer"]["save_dir"])

        experienceName = self._configuration["name"]
        if runId is None:
            runId = datetime.now().strftime(r"%m$d_%H%M%S")
        self._saveDirectory = saveDirectory / "models" / experienceName / runId
        self._logDirectory = saveDirectory / "log" / experienceName / runId

        # Make directory for saving checkpoints and log
        existOk = (runId == "")
        self._saveDirectory.mkdir(parents=True, exist_ok=existOk)
        self._logDirectory.mkdir(parents=True, exist_ok=existOk)

        # Save updated configuration file to the checkpoint directory
        write_json(self._configuration, self._saveDirectory / "config.json")

        # Configure Logging Module
        setup_logging(self._logDirectory)
        self.loggingLevels = {
                                0: logging.WARNING,
                                1: logging.INFO,
                                2: logging.DEBUG
                            }

    @classmethod
    def from_args(cls, args, options=""):
        """
        """
        for individualOption in options:
            args.add_argument(*individualOption.flags, default=None, type=individualOption.type)

        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device

        if args.resume is not None:
            resumePath = Path(args.resume)
            configurationFileName = resumePath.parent / "config.json"
        else:
            assert args.config is not None, "Configuration file needs to be specified. For example: Add '-c config.json'"
            resumePath = None
            configurationFileName = Path(args.config)

        configuration = read_json(configurationFileName)

        if args.config and resumePath:
            configuration.update(read_json(args.config))

        modification = {individualOption.target : getattr(args, _get_opt_name(individualOption.flags)) for individualOption in options}

        return cls(configuration, resumePath, modification)


    def initialize_object(self, name, module, *args, **kwargs):
        """
        """
        moduleName = self[name]["type"]
        moduleArguments = dict(self[name]["args"])
        assert all([k not in moduleArguments for k in kwargs]), "Overwriting kwargs given in configuration file is not allowed"
        moduleArguments.update(kwargs)
        return getattr(module, moduleName)(*args, **moduleArguments)

    def initialize_function(self, name, module, *args, **kwargs):
        """
        """
        moduleName = self[name]["type"]
        moduleArguments = dict(self[name]["args"])
        assert all([k not in moduleArguments for k in kwargs]), "Overwriting kwargs given in configuration file is not allowed"
        moduleArguments.update(kwargs)
        return partial(getattr(module, moduleName), *args, **moduleArguments)

    def __getitem__(self, name):
        return self._configuration[name]

    def get_logger(self, name, verbosity=2):
        assert verbosity in self.loggingLevels, "Verbosity Option {} is invalid. Valid Options are {}".format(verbosity, self.loggingLevels.keys())
        logger = logging.getLogger(name)
        logger.setLevel(self.loggingLevels[verbosity])
        return logger

    @property
    def config(self):
        return self._configuration

    @property
    def save_dir(self):
        return self._saveDirectory

    @property
    def log_directory(self):
        return self._logDirectory

def _update_config(configuration, modification):
    """
    """
    if modification is None:
        return configuration

    for key, value in modification.items():
        if value is not None:
            _set_by_path(configuration, key, value)
    return configuration

def _get_opt_name(flags):
    """
    """
    for individualFlag in flags:
        if individualFlag.startswith("--"):
            return individualFlag.replace("--", "")
    return flags[0].replace("--", "")

def _set_by_path(tree, keys, value):
    """
    """
    keys = keys.split(";")
    _get_by_path(tree, keys[:-1])[keys[-1]] = value

def _get_by_path(tree, keys):
    """
    """
    return reduce(getitem, keys, tree)
