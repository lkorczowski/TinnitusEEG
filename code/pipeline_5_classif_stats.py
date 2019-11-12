"""
Analysis of the results of pipeline_4_classif

:Authors:
    Louis Korczowski <louis.korczowski@gmail.com>

:history:
    | v0.1 2019-11-11 pipeline creation
"""

    # execution section
if __name__ == '__main__':
    ## ==============================================================================
    # IMPORTS
    # %%============================================================================

    import os, sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    import zeta

    # import matplotlib.pyplot as plt
    import mne
    import numpy as np
    import socket
    from itertools import compress
    import matplotlib.pyplot as plt
    import pickle
    import sklearn
    import time
    import pandas as pd

    # ==============================================================================
    # PATHS & CONFIG
    # %%============================================================================
    """" CONFIGURE PATHS 
    This section will be integrate into zeta module with a use of 
    configuration.py file prepared for each computer."""
    # TODO: integrate this section into a module or a yalm file

    datasetnames = ["Tinnitus_EEG","NormativeDB"]

    resultsID = 'pipeline_1_test'  # set ID for output directory (will remplace any former results with same ID)
    ForceSave = False  # if set True, will overwrite previous results in Pickle
    SaveFig = False  # if set True, will overwrite previous figure in folder  resultsID

    verbose = 'ERROR'

    data_dir, output_dir = zeta.configuration.load_directories()
    fig_dir = output_dir + os.path.sep + resultsID + os.path.sep  # pipeline output directory
    zeta.util.mkdir(fig_dir)  # create results directory if needed

    pd_results = pd.read_csv(os.path.join(output_dir, "results_classif.csv"))
