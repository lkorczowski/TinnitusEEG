#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility module for Zeta Module

@author: Louis Korczowski
"""

import pickle
import os


def save(variable,
         filepath,
         ForceSave=True):
    """save a variable into binary only if asked
    """
    if ForceSave:
        pickle.dump(variable, open(filepath, 'wb'))


def mkdir(target_dir):
    """Check if target dir exists and create it if needed

    TODO: populate this function for the Zeta module purpose (e.g. return error code if data are not computed)
    """

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

