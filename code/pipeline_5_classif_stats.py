"""
Analysis of the results of pipeline_4_classif

:Authors:
    Louis Korczowski <louis.korczowski@gmail.com>

:history:
    | v1.0 2019-11-12 from pipeline_4 auc=0.927 to auc 0.932 with adaboost
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

    pd_results = pd.read_csv(os.path.join(output_dir,resultsID, "results_classif.csv"))

    subjects = pd_results["subject"].unique()
    pipeline_nb = pd_results["pipeline_nb"].unique()

    # create a new dataset from different pipelines
    y_pip = []
    data_pip = []
    acc_pip = []
    roc_auc = []

    plt.close("all")
    for indp in pipeline_nb:
        pip_name = pd_results.loc[(pd_results["pipeline_nb"] == indp)]["pipeline"].values[0]
        data_pip.append(pd_results.loc[(pd_results["pipeline_nb"] == indp)]["predicted"].values)
        y_pip.append(pd_results.loc[(pd_results["pipeline_nb"] == indp)]["target"].values)
        #acc.append(sklearn.metrics.accuracy_score(y_pip[-1],data_pip[-1]))

        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_pip[-1], data_pip[-1])

        roc_auc.append(sklearn.metrics.auc(fpr, tpr))
        print("Area under the ROC curve for pipeline %i : %f" % (indp, roc_auc[-1]))

        ####################################
        # The optimal cut off would be where tpr is high and fpr is low
        # tpr - (1-fpr) is zero or near to zero is the optimal cut off point
        ####################################
        i = np.arange(len(tpr))  # index for df
        roc = pd.DataFrame(
            {'fpr': pd.Series(fpr, index=i), 'tpr': pd.Series(tpr, index=i), '1-fpr': pd.Series(1 - fpr, index=i),
             'tf': pd.Series(tpr - (1 - fpr), index=i), 'thresholds': pd.Series(thresholds, index=i)})
        print(roc.ix[(roc.tf - 0).abs().argsort()[:1]])
        best_threshold = roc.ix[(roc.tf - 0).abs().argsort()[:1]]["thresholds"].values[0]
        acc = []
        for thr in thresholds:
            acc.append(sklearn.metrics.accuracy_score(y_pip[-1],(data_pip[-1]>thr).astype(int)))
        acc_pip.append(max(acc))
        # Plot tpr vs 1-fpr
        fig, ax = plt.subplots()
        plt.plot(roc["thresholds"], roc['tpr'])
        plt.plot(roc["thresholds"], roc['1-fpr'], color='red')
        plt.xlabel('Threshold')
        plt.ylabel('Rate')
        plt.legend(["tpr", "1-fpr"])
        plt.title(pip_name + str(indp) + ': True and False Positive Rate, auc %.3f, acc %.3f'%(roc_auc[-1],acc_pip[-1]))
        plt.savefig(os.path.join(output_dir,resultsID,"auc_curve"  + pip_name + str(indp)))


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