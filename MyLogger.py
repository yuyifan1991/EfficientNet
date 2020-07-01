# -*- coding: utf-8 -*-

import logging
import os
from logging.handlers import TimedRotatingFileHandler
from logging.handlers import RotatingFileHandler

def get_logger_TimeRotating(log_filename, when, interval, backupCount, level):
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_filename = os.path.join(log_dir, log_filename)
    logger = logging.getLogger("CLASSIFY_TimeRotating")
    logger.setLevel(level)
    handler = TimedRotatingFileHandler(log_filename, when=when, interval=interval, backupCount=backupCount)
    datefmt = '%Y-%m-%d %H:%M:%S'
    format_str = '%(asctime)s %(levelname)s %(message)s '
    formatter = logging.Formatter(format_str, datefmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def get_logger_Rotating(log_filename, maxBytes, backupCount, level):
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_filename = os.path.join(log_dir, log_filename)
    logger = logging.getLogger("CLASSIFY_Rotating")
    logger.setLevel(level)
    handler = RotatingFileHandler(filename=log_filename, maxBytes=maxBytes, backupCount=backupCount)
    datefmt = '%Y-%m-%d %H:%M:%S'
    format_str = '%(asctime)s %(levelname)s %(message)s '
    formatter = logging.Formatter(format_str, datefmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def get_logger_Rotating_feedback(log_filename, maxBytes, backupCount, level):
    log_dir = "feedbacks"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_filename = os.path.join(log_dir, log_filename)
    logger = logging.getLogger("CLASSIFY_Rotating_feedback")
    logger.setLevel(level)
    handler = RotatingFileHandler(filename=log_filename, maxBytes=maxBytes, backupCount=backupCount)
    datefmt = '%Y-%m-%d %H:%M:%S'
    format_str = '%(asctime)s %(levelname)s %(message)s '
    formatter = logging.Formatter(format_str, datefmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
