#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility module for Zeta Module

@author: Louis Korczowski
"""

import pickle

def save(variable,
         filepath,
         ForceSave=True):
    """save a variable into binary only if asked
    """
    if ForceSave:
        pickle.dump(variable, open(filepath, 'wb'))
    

