#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Zeta Configuration Module
- Prepare machine-specific variable
- Load analysis-specific arguments for other functions
"""
import socket
import os


def load_directories():
    """ Load machine specific directories (need to be added manually and having the proper read-only or
    read/write access)

    Returns
    ----------
    data_dir : str
        root folder containing all the datasets (read only) and should by structure as following

        data_dir
        |
        +---dataset_name
        |   |   read_only_configuration_file (e.g. electrodes list, sample rate, readme, etc.)
        |   +---condition_name (optional)
        |       +---subject_name

    output_dir : str
        root folder for the figures and pickle files (read-write)

        output_dir
        |
        +---dataset_name
        |    +---test_name
            |   |   read-write_configuration_file (e.g. preprocessing steps, cut-off-frequencies, )
            |   +---subject_name
            |           subject_specific_figure_name.png
            |           subject_specific_result_name.pickle
            |   +---group-level_name
            |           group_specific_figure_name.png
            |           group_specific_result_name.pickle
            |

    """
    data_dir = []
    output_dir = []
    config_id = socket.gethostname()
    if config_id == 'your_machine_name':
        print("not configured")
    elif config_id == 'Crimson-Box':
        #os.chdir("F:\\git\\TinnitusEEG\\code")
        data_dir = os.path.join("F:\\", "data", 'Zeta')
        output_dir = os.path.join("D:\\", "GoogleDrive", "Zeta Technologies", "Zeta_shared", "results")
    elif config_id == 'MacBook-Pro-de-Louis.local' or 'MBP-de-Louis':
        #os.chdir("/Volumes/Ext/git/TinnitusEEG/code")
        data_dir = os.path.join("/Volumes/Ext/", "data", 'Zeta')
        output_dir = '/Users/louis/Google Drive/Zeta Technologies/Zeta_shared/results'
    else:
        print('configuration.py not recognize, please add the path of your git directories')

    return data_dir, output_dir
