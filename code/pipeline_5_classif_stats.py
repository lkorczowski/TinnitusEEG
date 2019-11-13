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

    subjects = pd_results["subject"].unique()
    pipeline_nb = pd_results["pipeline_nb"].unique()

    # create a new dataset from different pipelines
    data = []
    y = []
    for subject in subjects:
        data_sub= []
        for indp in pipeline_nb:
            data_sub.append(pd_results.loc[(pd_results["subject"] == subject) & (pd_results["pipeline_nb"] == indp)]["predicted"].values)
        if len(data_sub[0]) == 50:
            data.append(np.array(data_sub))
            assert np.unique(pd_results.loc[(pd_results["subject"] == subject)]["target"].values).size == 1
            y.append(np.unique(pd_results.loc[(pd_results["subject"] == subject)]["target"].values)[0])
        else:
            print("unconsistant number of epochs for ML %i"%len(data_sub[0]))

    data = np.array(data)

    pipeline = zeta.pipelines.CreatesFeatsPipeline("vot_ADA")

    n_splits = 5
    outer_cv = sklearn.model_selection.StratifiedKFold(n_splits=n_splits)

    time1 = time.time()
    score = sklearn.model_selection.cross_val_score(pipeline, X=data.reshape(data.shape[0],-1), y=y, cv=outer_cv,scoring='roc_auc')
    time2 = time.time()
    bestauc = pd_results["auc"].values.max()
    print("new auc %.3f (+/- %.3f std) versus best auc %.3f" % (np.mean(score),np.std(score),bestauc))