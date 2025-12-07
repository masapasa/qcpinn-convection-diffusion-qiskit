import logging
from datetime import datetime
import os
import numpy as np
class Logging:
    def __init__(self, log_path, experiment_name=None, source_file=None):
        self.log_path = log_path
        self.experiment_name = experiment_name
        self._create_output_dir(source_file)
        self._create_logger()
    def _create_output_dir(self, source_path=None):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        if self.experiment_name is not None:
            timestamp = f"{timestamp}_{self.experiment_name}"
        try:
            self.output_dir = os.path.join(self.log_path, timestamp)
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError as error:
            print(f"Error: {error.strerror}")
    def _create_logger(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        logging.basicConfig(filemode="w")
        fh = logging.FileHandler(f"{self.output_dir}/output.log")
        self.logger.addHandler(fh)
    def get_output_dir(self):
        return self.output_dir
    def print(self, *args):
        for arg in args:
            if len(args) == 1:
                self.logger.info(arg)
            elif arg != args[-1]:
                for handler in self.logger.handlers:
                    handler.terminator = ""
                if (
                    type(arg) == float
                    or type(arg) == np.float64
                    or type(arg) == np.float32
                ):
                    self.logger.info("%.4e" % (arg))
                else:
                    self.logger.info(arg)
            else:
                for handler in self.logger.handlers:
                    handler.terminator = "\n"
                if (
                    type(arg) == float
                    or type(arg) == np.float64
                    or type(arg) == np.float32
                ):
                    self.logger.info("%.4e" % (arg))
                else:
                    self.logger.info(arg)