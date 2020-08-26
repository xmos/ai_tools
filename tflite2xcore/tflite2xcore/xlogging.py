# Copyright (c) 2018-2020, XMOS Ltd, All rights reserved


import logging
from logging import NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL


XDEBUG = DEBUG - 1

logging.addLevelName(XDEBUG, "XDEBUG")


def getLogger(*args, **kwargs):
    logger = logging.getLogger(*args, *kwargs)
    logger.xdebug = lambda *args, **kwargs: logger.log(XDEBUG, *args, **kwargs)
    return logger


class LoggingContext:
    def __init__(self, logger, level=None, handler=None, close=True):
        self.logger = logger
        self.level = level
        self.handler = handler
        self.close = close

    def __enter__(self):
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        if self.handler:
            self.logger.addHandler(self.handler)

    def __exit__(self, exception_type, exception_value, traceback):
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()

