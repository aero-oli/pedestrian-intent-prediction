# Implementation of the class TensorBoardWriter 

import importlib
from datetime import datetime

class TensorBoardWriter:
    """
    Class implementation for Tensor Board Writer.
    """
    def __init__(self, logDirectory, logger, enabled):
        """
        Method to initialize an object of type TensorBoardWriter.

        Parameters
        ----------
        self            : TensorBoardWriter
                          Instance of the class
        logDirectory    : str
                          Directory of the logger
        logger          : Logger
                          Logger instance to log information on the TensorBoard
        enabled         : bool
                          Informs if the Tensor Board Writer is enabled

        Returns
        -------
        self    : TensorBoardWriter
                  Initialized object of class TensorBoardWriter
        """
        # Initialize parameters like writer and selected module
        self.writer = None
        self.selectedModule = ""

        # Import required modules if TensorBoardWriter is enabled
        if enabled:
            logDirectory = str(logDirectory)
            importSuccessful = False
            for module in ["torch.utils.tensorboard", "tensorboardX"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(logDirectory)
                    importSuccessful = True
                    break
                except ImportError:
                    importSuccessful = False
                self.selectedModule = module

            if not importSuccessful:
                message = "[WARNING] Visualization (i.e. Tensorboard) is configured to use, but not installed on this machine. " \
                    "Please install TensorboardX with 'pip install tensorboardx' and upgrade PyTorch to version >= 1.1 to use" \
                    "'torch.utils.tensorboard' or turn off the option in 'config.json' file." 
                logger.warning(message)

        # Initialize parameters like step, mode, tensorboardWriterFunctions, tagModeExceptions and timer
        self.step = 0
        self.mode = ""
        self.tensorboardWriterFunctions =   { 
                                                "add_scalar", "add_scalars", "add_image", "add_images", "add_audio", 
                                                "add_text", "add_histogram", "add_pr_curve", "add_embedding"
                                            }        
        self.tagModeExceptions = {"add_histogram", "add_embedding"}
        self.timer = datetime.now()

    def __getattr__(self, name):
        """
        Method to get an attribute of the class.
        If the visualization is configured to use, the method returns an addData() method of tensorboard with additional information (step, tag) added.
        Else the method returns a blank function handle that does nothing.

        Parameters
        ----------
        self    : TensorBoardWriter
                  Instance of the class
        name    : str
                  Name of the attribute

        Returns
        -------
        function    : Callable
                      Function handle that adds data in case the visualization is configured to use, else blank
        """
        # Evaluate if visualization is configured and return a function handle
        if name in self.tensorboardWriterFunctions:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    if name not in self.tagModeExceptions:
                        tag = '{}/{}'.format(tag, self.mode)
                    add_data(tag, data, self.step, *args, **kwargs)
            return wrapper
        else:
            try:
                attribute = object.__getattr__(name)
            except AttributeError:
                raise AttributeError('Type object {} has no attribute {}'.format(self.selectedModule, name))        
            return attribute


    def set_step(self, step, mode='train'):
        """
        Method to set step for the Tensorboard writer.

        Parameters
        ----------
        self    : TensorBoardWriter
                  Instance of the class
        step    : int
                  Number of steps
        mode    : str
                  Mode of the steps (train/test)

        Returns
        -------
        None
        """
        # Initialize parameters like mode and step
        self.mode = mode
        self.step = step

        # Set step for the TensorBoardWriter
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar("steps_per_second", 1/duration.total_seconds())
            self.timer = datetime.now()