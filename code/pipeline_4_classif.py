# -*- coding: utf-8 -*-
"""
Classify the data of a given dataset (intra or inter-subject level).

:Authors:
    Louis Korczowski <louis.korczowski@gmail.com>

:history:
    | v3.0 2019-11-11 Tinnitus_EEG versus NormativeDB
    | v2.5 2019-10-29 Multi-pipeline integration and results reporting
    | v2.0 2019-10-10 Distress2010 versus NormativeDB: AUC=0.88
    | v1.5 2019-10-07 adding multi-dataset support ("Distress2010", "NormativeDB", "Tinnitus_EEG" added)
    | v1.4 2019-09-25 integrated groups for the crossvalidation
    | v1.3 2019-09-25 inter-subject classification working (AUC=0.55 in low versus high stimuli dataset)
    | v1.2 2019-09-24 inter-subject pipelines deployed
    | v1.1 2019-09-23 refactored: util, datasets and configuration modules + n_minimum_epochs_to_classif
    | v1.0 2019-09-17 minor integrations and documentation
    | v0.5 2019-09-11 working xdawn+LR classification on lowVhigh dataset (AUC>0.9 for most subjects)
    | v0.1 2019-09-05 pipeline creation, integration with zeta.pipeline module

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

    ## ----------------------------
    # pipeline_4 specific configuration & preload data
    # %%---------------------------

    n_minimum_epochs_to_classif=1  # assess the minimal number of extracted epochs per class to perform a valid cross-validation
    max_epochs = 50 # WARNING: just to speed up computation
    operations_to_apply = dict(
        epoching=1,
        GFP=0,  # Global Field Power
        TFR=0,  # Time-Frequency Response for each epoch
        TFR_av=0,  # Time-Frequency Response Averaging
        TFR_stats=0,  # Compute inter-trials statistics on TFR
        cla_ERP_TS_LR=0,
        inter_subject=1
    )

    rejected_subjects = []
    all_X = []    # for inter-subject classification
    all_y = []    # for inter-subject classification
    groups = []   # for cross-validation segmentation
    subjects_names = [] # to track in inter_subject

    #==============================================================================
    # DATASET LOOP
    #%%============================================================================
    for datasetname in datasetnames:
        #==============================================================================
        # META DATA LOADING
        #%%============================================================================

        subjects = list(zeta.data.datasets.get_subjects_info(data_dir, datasetname).keys())
        subjects_error=[]
        if datasetname is "Distress2010":
            # TODO: can't load those subjects
            subjects.remove("4BAUWENS_SIMONNE_ZONDER_STIM_VIJF_OP_TIEN")
            subjects.remove("2REYLANDT_ANNICK")
        if datasetname is "Tinnitus_EEG":
            # TODO: can't load those subjects
            subjects.remove("VAN_DEN_BOSSCHE_INGRID")
            subjects.remove("VOUNCKX_MARCEL")
            subjects.remove("SCHRUERS_MARC_ZONDER_TIN")


        ## =============================================================================
        # PROCESSING LOOP
        # %%============================================================================
        for inds, subject in enumerate(subjects): # [list(subjects)[1]]:  #   #
            # ----------------------------
            # RAW DATA LOADING AND PREPROCESSING
            # %%--------------------------
            try:
                fig_dir_sub = fig_dir + subject + os.path.sep
                zeta.util.mkdir(fig_dir_sub)  # create results directory if needed
                # load raw data
                raw_0, raw_1, events0, events1 = zeta.data.datasets.get_raw(data_dir, datasetname, subject)

                print(subject + ": data loaded")
            except IndexError:
                print(subject + ": ERROR data NOT loaded")
                subjects_error.append(subject)
                continue

            ## ----------------------------
            # epoching
            # %%--------------------------
            if operations_to_apply["epoching"]:
                #            try:
                plt.close('all')

                if datasetname is "raw_clean_32":

                    # extract epochs
                    event_id0, event_id1, tmin, tmax = {'low': 0}, {'high': 1}, 5.6, 7.6
                    baseline = (None, 5.6)
                    reject = dict(eeg=200)
                    epochs0 = mne.Epochs(raw_0, events0, event_id0, tmin, tmax, baseline=baseline,
                                         reject=reject, preload=True, reject_by_annotation=0,
                                         verbose=verbose)

                    epochs1 = mne.Epochs(raw_1, events1, event_id1, tmin, tmax, baseline=baseline,
                                         reject=reject, preload=True, reject_by_annotation=0,
                                         verbose=verbose)

                    if (epochs0.__len__()<n_minimum_epochs_to_classif
                            or epochs1.__len__()<n_minimum_epochs_to_classif):
                        rejected_subjects.append(subject)
                        print(subject + ": not enough valid epochs. Count: (%i and %i)" %(epochs0.__len__(), epochs1.__len__()))

                else:
                    # extract epochs
                    if datasetname is "Distress2010":
                        event_id0 = None
                    if datasetname is "NormativeDB":
                        event_id0 = {"Control": 0}
                    else:
                        event_id0 = None

                    event_id1 = None
                    tmin, tmax = 0, 1.5
                    baseline = (None, 0)
                    reject = dict(eeg=200)
                    epochs0 = mne.Epochs(raw_0, events0, event_id0, tmin, tmax, baseline=baseline,
                                         reject=reject, preload=True, reject_by_annotation=0,
                                         verbose=verbose)

                    epochs0 = epochs0[0:min(epochs0.__len__(),max_epochs)]
                    print("warning: only %i epochs selected" %min(epochs0.__len__(),max_epochs))
                    epochs1 = None

                print(subject + ": epoching done")


            ## ----------------------------
            # Classification 1
            # %%--------------------------
            if (subject in rejected_subjects) or not operations_to_apply["epoching"]:
                print(subject + ": skip classification")
            else:
                if datasetname in ["raw_clean_32"]:
                    epochs = mne.concatenate_epochs([epochs0, epochs1])
                elif datasetname in ["NormativeDB","Distress2010","Tinnitus_EEG"]:
                    epochs = mne.concatenate_epochs([epochs0])


                if operations_to_apply["cla_ERP_TS_LR"]:
                    # %% 1
                    PipelineTitle = 'cla_ERP_TS_LR'
                    PipelineNb = 0
                    doGridSearch = 0
                    n_splits=5

                    # ----------------------------
                    # step0: Prepare pipeline (hyperparameters & CV)
                    # ----------------------------
                    inner_cv = sklearn.model_selection.StratifiedKFold(n_splits=n_splits, random_state=42)
                    outer_cv = sklearn.model_selection.StratifiedKFold(n_splits=n_splits, random_state=42)


                    hyperparameters = {
                        # 'preproc__tmin': [5.6],
                        # 'preproc__tmax': [7.6],
                        # 'preproc__epochstmin': [epochs.tmin],  # static
                        # 'preproc__epochsinfo': [epochs.info],  # static
                        # 'preproc__baseline': [(None, 5.6)],
                        # 'preproc__filters': [([1, 30],)],
                        'xdawn__estimator': ['lwf'],
                        'xdawn__xdawn_estimator': ['lwf'],
                        'xdawn__nfilter': [12],
                        # 'xdawn__bins':[[x for x in [0,20,40,60,80,100]]],
                        # 'LASSO__cv': [inner_cv],  # static
                        # 'LASSO__random_state': [42],  # static
                        # 'LASSO__max_iter': [250]  # static
                        'lr__solver': ['lbfgs'],
                    }

                    init_params = {}
                    for item in hyperparameters:
                        init_params[item] = hyperparameters[item][0]

                    # ----------------------------
                    # step1: Prepare pipeline & inputs
                    # ----------------------------

                    X = []
                    tmp = epochs.copy().drop(epochs.events[:, 2] < 0)  # remove bad epochs
                    X = tmp.get_data()
                    y = tmp.events[:, 2]

                    # check cross validation
                    fig, ax = plt.subplots()
                    zeta.viz.classif.plot_cv_indices(outer_cv,X,y,ax=ax)

                    pipeline = zeta.pipelines.CreatesFeatsPipeline(PipelineTitle, init_params=init_params)
                    # ----------------------------
                    # step2: GridSearch
                    # ----------------------------

                    if doGridSearch:
                        pipeline = sklearn.model_selection.GridSearchCV(pipeline, hyperparameters, cv=inner_cv, scoring='auc')

                    # ----------------------------
                    # step3: Predict in CV
                    # ----------------------------

                    time1 = time.time()
                    predicted = sklearn.model_selection.cross_val_predict(pipeline, X=X, y=y, cv=outer_cv)
                    time2 = time.time()

                    # ----------------------------
                    # step4: PRINT RESULTS
                    # ----------------------------
                    print(PipelineTitle + ' in %f' % (time2 - time1) + 's')
                    zeta.viz.classif.ShowClassificationResults(y, predicted, PipelineTitle=PipelineTitle, ax=None)

                    # ----------------------------
                    # step5: Save
                    # ----------------------------
                    pipelineERP = pipeline
                    predictedERP = predicted

                if operations_to_apply["inter_subject"]:
                    X = []
                    tmp = epochs.copy().drop(epochs.events[:, 2] < 0)  # remove bad epochs
                    X = tmp.get_data()
                    y = tmp.events[:, 2]
                    all_X.append(X)
                    all_y.append(y)
                    subjects_names.append(subject)

            print("Subject %i / %i DONE" %(inds+1, len(subjects)))

    if operations_to_apply["inter_subject"]:
        import sklearn.metrics as metrics
        pd_results=pd.DataFrame()

        PipelineNb = -1
        ##0
        PipelineTitle="cla_MDM"
        PipelineNb = PipelineNb + 1
        doGridSearch = 0
        n_splits = 5
        # ----------------------------
        # step-1: organize group-based data
        # ----------------------------
        X = np.concatenate(all_X)
        y = np.concatenate(all_y)

        # ----------------------------
        # step0: Prepare pipeline (hyperparameters & CV)
        # ----------------------------
        groups=[]
        groups_names = []
        for i in range(len(all_y)):
            n_epochs = all_y[i].shape[0]
            groups = np.concatenate((groups, [i] * n_epochs))  # assign each subject to a specific group
            groups_names = np.concatenate((groups_names, [subjects_names[i]] * n_epochs))  # name group

        inner_cv = sklearn.model_selection.GroupKFold(n_splits=n_splits)  # KFOLD with respect to the user
        outer_cv = sklearn.model_selection.GroupKFold(n_splits=n_splits)


        # hyperparameters kept here for compatibility
        # TODO: use function to integrate hyperparameters more easily using get_params and homemade set_params

        hyperparameters = {}

        init_params = {}
        for item in hyperparameters:
            init_params[item] = hyperparameters[item][0]

        # ----------------------------
        # step1: Prepare pipeline & inputs
        # ----------------------------

        # check cross validation
        fig, ax = plt.subplots()
        zeta.viz.classif.plot_cv_indices(outer_cv, X, y, ax=ax,group=groups)
        import pyriemann
        pipeline = zeta.pipelines.CreatesFeatsPipeline(PipelineTitle, init_params=init_params)

        # ----------------------------
        # step2: GridSearch
        # ----------------------------

        if doGridSearch:
            pipeline = sklearn.model_selection.GridSearchCV(pipeline, hyperparameters, cv=inner_cv, scoring='auc')

        # ----------------------------
        # step3: Predict in CV
        # ----------------------------

        time1 = time.time()
        predicted = sklearn.model_selection.cross_val_predict(pipeline, X=X, y=y, cv=outer_cv, groups=groups, method = 'transform')
        predicted = -np.diff(predicted)
        time2 = time.time()

        # ----------------------------
        # step4: PRINT RESULTS
        # ----------------------------
        print(PipelineTitle + ' in %f' % (time2 - time1) + 's')

        # TODO: may be necessary to explicit the CV loop as scoring on whole cv is biaised and doesn't inform to the
        # TODO: behaviour of the classification in CV
        # report = zeta.viz.classif.ShowClassificationResults(y, predicted, PipelineTitle=PipelineTitle, ax=None)
        auc = metrics.roc_auc_score(y, predicted)
        print("auc %.3f"%auc)

        # ----------------------------
        # step5: Save
        # ----------------------------
        pipelineERP = pipeline
        predictedERP = predicted

        resultsDict = {"subject": groups_names, "target": y, "predicted": np.squeeze(predicted), "pipeline_nb": PipelineNb, "pipeline": PipelineTitle, "auc": auc}
        pd_results = pd_results.append(pd.DataFrame(resultsDict))

        ##1
        PipelineTitle = "cla_CSP_MDM"
        PipelineNb = PipelineNb + 1
        doGridSearch = 0
        n_splits = 5
        # ----------------------------
        # step-1: organize group-based data
        # ----------------------------
        X = np.concatenate(all_X)
        y = np.concatenate(all_y)

        # ----------------------------
        # step0: Prepare pipeline (hyperparameters & CV)
        # ----------------------------
        groups = []
        groups_names = []
        for i in range(len(all_y)):
            n_epochs = all_y[i].shape[0]
            groups = np.concatenate((groups, [i] * n_epochs))  # assign each subject to a specific group
            groups_names = np.concatenate((groups_names, [subjects_names[i]] * n_epochs))  # name group

        inner_cv = sklearn.model_selection.GroupKFold(n_splits=n_splits)  # KFOLD with respect to the user
        outer_cv = sklearn.model_selection.GroupKFold(n_splits=n_splits)

        # hyperparameters kept here for compatibility
        # TODO: use function to integrate hyperparameters more easily using get_params and homemade set_params

        hyperparameters = {}

        init_params = {}
        for item in hyperparameters:
            init_params[item] = hyperparameters[item][0]

        # ----------------------------
        # step1: Prepare pipeline & inputs
        # ----------------------------

        # check cross validation
        fig, ax = plt.subplots()
        zeta.viz.classif.plot_cv_indices(outer_cv, X, y, ax=ax, group=groups)
        import pyriemann

        pipeline = zeta.pipelines.CreatesFeatsPipeline(PipelineTitle, init_params=init_params)

        # ----------------------------
        # step2: GridSearch
        # ----------------------------

        if doGridSearch:
            pipeline = sklearn.model_selection.GridSearchCV(pipeline, hyperparameters, cv=inner_cv, scoring='auc')

        # ----------------------------
        # step3: Predict in CV
        # ----------------------------

        time1 = time.time()
        predicted = sklearn.model_selection.cross_val_predict(pipeline, X=X, y=y, cv=outer_cv, groups=groups,
                                                              method='transform')
        predicted = -np.diff(predicted)
        time2 = time.time()

        # ----------------------------
        # step4: PRINT RESULTS
        # ----------------------------
        print(PipelineTitle + ' in %f' % (time2 - time1) + 's')

        # TODO: may be necessary to explicit the CV loop as scoring on whole cv is biaised and doesn't inform to the
        # TODO: behaviour of the classification in CV
        # report = zeta.viz.classif.ShowClassificationResults(y, predicted, PipelineTitle=PipelineTitle, ax=None)
        auc = metrics.roc_auc_score(y, predicted)
        print("auc %.3f"%auc)
        # ----------------------------
        # step5: Save
        # ----------------------------
        pipelineERP = pipeline
        predictedERP = predicted

        resultsDict = {"subject": groups_names, "target": y, "predicted": np.squeeze(predicted),
                       "pipeline_nb": PipelineNb, "pipeline": PipelineTitle, "auc": auc}
        pd_results = pd_results.append(pd.DataFrame(resultsDict))

        ##2
        PipelineTitle = "cla_CSP_LR"
        PipelineNb = PipelineNb + 1
        doGridSearch = 1
        n_splits = 5
        # ----------------------------
        # step-1: organize group-based data
        # ----------------------------
        X = np.concatenate(all_X)
        y = np.concatenate(all_y)

        # ----------------------------
        # step0: Prepare pipeline (hyperparameters & CV)
        # ----------------------------
        groups = []
        groups_names = []
        for i in range(len(all_y)):
            n_epochs = all_y[i].shape[0]
            groups = np.concatenate((groups, [i] * n_epochs))  # assign each subject to a specific group
            groups_names = np.concatenate((groups_names, [subjects_names[i]] * n_epochs))  # name group

        inner_cv = sklearn.model_selection.StratifiedKFold(n_splits=n_splits)  # KFOLD with respect to the user
        outer_cv = sklearn.model_selection.GroupKFold(n_splits=n_splits)

        # hyperparameters kept here for compatibility
        # TODO: use function to integrate hyperparameters more easily using get_params and homemade set_params

        hyperparameters = {"CSP__nfilter":[4,6,8,10,12]}

        init_params = {}
        for item in hyperparameters:
            init_params[item] = hyperparameters[item][0]

        # ----------------------------
        # step1: Prepare pipeline & inputs
        # ----------------------------

        # check cross validation
        fig, ax = plt.subplots()
        zeta.viz.classif.plot_cv_indices(outer_cv, X, y, ax=ax, group=groups)
        import pyriemann

        pipeline = zeta.pipelines.CreatesFeatsPipeline(PipelineTitle, init_params=init_params)

        # ----------------------------
        # step2: GridSearch
        # ----------------------------

        if doGridSearch:
            pipeline = sklearn.model_selection.GridSearchCV(pipeline, hyperparameters, cv=inner_cv, scoring='roc_auc')

        # ----------------------------
        # step3: Predict in CV
        # ----------------------------

        time1 = time.time()
        predicted = sklearn.model_selection.cross_val_predict(pipeline, X=X, y=y, cv=outer_cv, groups=groups,method="predict_log_proba")
        predicted = predicted[:,1]
        time2 = time.time()

        # ----------------------------
        # step4: PRINT RESULTS
        # ----------------------------
        print(PipelineTitle + ' in %f' % (time2 - time1) + 's')

        # TODO: may be necessary to explicit the CV loop as scoring on whole cv is biaised and doesn't inform to the
        # TODO: behaviour of the classification in CV
        # report = zeta.viz.classif.ShowClassificationResults(y, predicted, PipelineTitle=PipelineTitle, ax=None)
        auc = metrics.roc_auc_score(y, predicted)
        print("auc %.3f"%auc)

        # ----------------------------
        # step5: Save
        # ----------------------------
        pipelineERP = pipeline
        predictedERP = predicted

        resultsDict = {"subject": groups_names, "target": y, "predicted": np.squeeze(predicted),
                       "pipeline_nb": PipelineNb, "pipeline": PipelineTitle, "auc": auc}
        pd_results = pd_results.append(pd.DataFrame(resultsDict))

        ##3
        PipelineTitle = "cla_CSP_LR"
        PipelineNb = PipelineNb + 1
        doGridSearch = 0
        n_splits = 5
        # ----------------------------
        # step-1: organize group-based data
        # ----------------------------
        X = np.concatenate(all_X)
        y = np.concatenate(all_y)

        # ----------------------------
        # step0: Prepare pipeline (hyperparameters & CV)
        # ----------------------------
        groups = []
        groups_names = []
        for i in range(len(all_y)):
            n_epochs = all_y[i].shape[0]
            groups = np.concatenate((groups, [i] * n_epochs))  # assign each subject to a specific group
            groups_names = np.concatenate((groups_names, [subjects_names[i]] * n_epochs))  # name group

        inner_cv = sklearn.model_selection.StratifiedKFold(n_splits=n_splits)  # KFOLD with respect to the user
        outer_cv = sklearn.model_selection.LeaveOneGroupOut()

        # hyperparameters kept here for compatibility
        # TODO: use function to integrate hyperparameters more easily using get_params and homemade set_params

        hyperparameters = {"CSP__nfilter":[4,6,8,10,12]}

        init_params = {}
        for item in hyperparameters:
            init_params[item] = hyperparameters[item][0]

        # ----------------------------
        # step1: Prepare pipeline & inputs
        # ----------------------------

        # check cross validation
        fig, ax = plt.subplots()
        zeta.viz.classif.plot_cv_indices(outer_cv, X, y, ax=ax, group=groups)
        import pyriemann

        pipeline = zeta.pipelines.CreatesFeatsPipeline(PipelineTitle, init_params=init_params)

        # ----------------------------
        # step2: GridSearch
        # ----------------------------

        if doGridSearch:
            pipeline = sklearn.model_selection.GridSearchCV(pipeline, hyperparameters, cv=inner_cv, scoring='roc_auc')

        # ----------------------------
        # step3: Predict in CV
        # ----------------------------

        time1 = time.time()
        predicted = sklearn.model_selection.cross_val_predict(pipeline, X=X, y=y, cv=outer_cv, groups=groups,method="predict_log_proba")
        predicted = predicted[:,1]
        time2 = time.time()

        # ----------------------------
        # step4: PRINT RESULTS
        # ----------------------------
        print(PipelineTitle + ' in %f' % (time2 - time1) + 's')

        # TODO: may be necessary to explicit the CV loop as scoring on whole cv is biaised and doesn't inform to the
        # TODO: behaviour of the classification in CV
        # report = zeta.viz.classif.ShowClassificationResults(y, predicted, PipelineTitle=PipelineTitle, ax=None)
        auc = metrics.roc_auc_score(y, predicted)
        print("auc %.3f"%auc)

        # ----------------------------
        # step5: Save
        # ----------------------------
        pipelineERP = pipeline
        predictedERP = predicted

        resultsDict = {"subject": groups_names, "target": y, "predicted": np.squeeze(predicted),
                       "pipeline_nb": PipelineNb, "pipeline": PipelineTitle, "auc": auc}
        pd_results = pd_results.append(pd.DataFrame(resultsDict))

        pd_results.to_csv(os.path.join(output_dir,"results_classif.csv"))

    print("Pipeline 4 DONE")
