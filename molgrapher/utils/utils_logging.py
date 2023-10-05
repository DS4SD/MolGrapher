#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# System
import os
import sys

# Logging
import logging

def set_logging(name, output_file):
    logging.basicConfig(
        filename = output_file, 
        filemode = 'a',
        format = "%(asctime)s - [%(levelname)s] - [PID %(process)d - TID %(thread)d] - %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    h = logging.StreamHandler(sys.stdout)
    logger.addHandler(h)
    return logger

def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)