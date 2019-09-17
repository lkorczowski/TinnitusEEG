# -*- coding: utf-8 -*-
"""
Vizualize data and performance event-related statistics

Input:
    - 
Output:
    - 

@author: louis.korczowski@gmail.com

PROPERTY OF Zeta Technologies
CONFIDENTIAL

last version: DRAFT v0.5 (2019-07-19)
history:
    | v0.71 2019-08-21 backup v0.5, pickle integration
    | v0.7 2019-07-27 group-level integration (CANCELED)
    | v0.6 2019-07-23 dataset integration (CANCELED)
    | v0.5 2019-07-19 correction of timing
    | v0.4 2019-07-17 TF statistical test and GFP
    | v0.3 2019-07-16 epoch extraction and TF vizualization
    | v0.2 2019-07-15 annotation to events convertion
    | v0.1 2019-07-11 pipeline creation

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

    # ==============================================================================
    # PATHS & CONFIG
    # %%============================================================================
    """" CONFIGURE PATHS 
    This section will be integrate into zeta module with a use of 
    config file prepared for each computer."""
    # TODO: integrate this section into a module or a yalm file

    resultsID = 'pipeline_1_test'  # set ID for output directory (will remplace any former results with same ID)
    ForceSave = False  # if set True, will overwrite previous results in Pickle
    SaveFig = False  # if set True, will overwrite previous figure in folder  resultsID

    operations_to_apply = dict(
        epoching=1,
        GFP=0,  # Global Field Power
        TFR=0,  # Time-Frequency Response for each epoch
        TFR_av=0,  # Time-Frequency Response Averaging
        TFR_stats=0,  # Compute inter-trials statistics on TFR
        cla_ERPCovMDM=1
    )
    verbose = 'ERROR'
    subject = 1
    patient = 2  # patient group (static for a given dataset)
    session = 9  # 6 = 1 old remplacer apres (session 'high')
    ses2 = 8  # (session 'low')
    datasetname = "raw_clean_32"

    # for automatic folder path use, add the elif: for your machine ID below
    configID = socket.gethostname()
    if configID == 'your_machine_name':
        print("not configured")
    elif configID == 'Crimson-Box':
        os.chdir("F:\\git\\TinnitusEEG\\code")
        data_dir = os.path.join("F:\\", "data", 'Zeta')
        fig_dir = os.path.join("D:\\", "GoogleDrive", "Zeta Technologies", "Zeta_shared", "results")
    elif configID == 'MacBook-Pro-de-Louis.local':
        os.chdir("/Volumes/Ext/git/TinnitusEEG/code")
        data_dir = os.path.join("/Volumes/Ext/", "data", 'Zeta')
        fig_dir = '/Users/louis/Google Drive/Zeta Technologies/Zeta_shared/results'
    else:
        print('config not recognize, please add the path of your git directories')
    # WARNING : "\\" is used for windows, "/" is used for unix (or 'os.path.sep')

    fig_dir = fig_dir + os.path.sep + resultsID + os.path.sep  # pipeline output directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)  # create results directory if needed

    """
    subjects=['patientPC', '5CA09', '05AY22', 'patient', '05GS16', '04MM25', 
          '1TG01', '05RP24', '3QO03', '5CD05', '5DN04', '05IN17', '05VP19', 
          'GS', '05MP21', '5DL08', '05BY20', '05DF18', '05MV11', '5NA09',
          'CQ', '5BB03', '05FV18', '5BY10', '04LK03', '04LM02', '05RM12',
          '2SN14', 'QE', '3NT07', '5GF07']  #obsolete, generated below
    # please note that 'patient_CR' is NOT a good name as the sparse with '_' is not
    # compatible with the other files. I recommand renamed it to 'patientCR' or 'CR'
    """

    # ==============================================================================
    # META DATA LOADING
    # %%============================================================================

    names = os.listdir(os.path.join(data_dir, datasetname, str(patient) + "_" + str(session)))
    names2 = os.listdir(os.path.join(data_dir, datasetname, str(patient) + "_" + str(ses2)))

    pat = []
    pat2 = []
    for name in names:
        # print name.split('_')[0]
        pat.append(name.split('_')[0])  # all subjects ID from names
    for name in names2:
        # print name.split('_')[0]
        pat2.append(name.split('_')[0])  # all subjects ID from names2

    cong = {}  # build of dictionnary of all session for each subject
    for name in names2:
        if pat.__contains__(name.split('_')[0]):
            if cong.keys().__contains__(name.split('_')[0]):
                cong[name.split('_')[0]].append(name)  # add file to the list
            else:
                cong[name.split('_')[0]] = [name]  # add first file to the list
    for name in names:
        if pat2.__contains__(name.split('_')[0]):
            cong[name.split('_')[0]].append(name)

    t = ["exacerb", "absente", "partielle", "totale"]
    subjects = cong.keys()

    # ==============================================================================
    # PROCESSING LOOP
    # %%============================================================================
    for subject in [list(subjects)[1]]:  # list(subjects): #
        # ----------------------------
        # RAW DATA LOADING AND PREPROCESSING
        # %%--------------------------
        fig_dir_sub = fig_dir + subject + os.path.sep
        if not os.path.exists(fig_dir_sub):
            os.makedirs(fig_dir_sub)  # create results directory if needed

        # load subject data
        """
        note here that the EEG data are in µV while MNE use V. Therefore scale 
        is with a 1e6 factor andit could cause a problem for non-linear related MNE
        analysing. I advice to apply a 1e-6 factor in the future to make sure that
        everything is working fine with mne.
        For classification, it is adviced to keep data in µV.
        """

        # clean loop variable
        runs = []
        labels = []
        events = []

        # load raw data
        for root, dirs, files in os.walk(os.path.join(data_dir, datasetname)):
            for file in files:
                if file.startswith(subject):
                    filepath = os.path.join(root, file)
                    runs.append(mne.io.read_raw_fif(filepath, verbose="ERROR"))  # load data
                    # events=mne.read_events(filepath)
                    labels.append(file.split('_')[1])

        eventscode = dict(zip(np.unique(runs[0]._annotations.description), [0, 1]))

        runs_0 = list(compress(runs, [x == 'low' for x in labels]))
        runs_1 = list(compress(runs, [x == 'high' for x in labels]))

        raw_0 = mne.concatenate_raws(runs_0)
        raw_1 = mne.concatenate_raws(runs_1)

        # rename table for event to annotations
        event_id0 = {'BAD_data': 0, 'bad EPOCH': 100, 'BAD boundary': 100, 'EDGE boundary': 100}

        # vizualization
        if SaveFig:
            scalings = dict(eeg=10e1)
            mne.viz.plot_raw(raw_0, scalings=scalings)

        timestamps = np.round(raw_0._annotations.onset * raw_0.info['sfreq']).astype(int)

        raw_0._annotations.duration
        event_id = {'BAD_data': 0, 'bad EPOCH': 100}

        labels = raw_0._annotations.description
        labels = np.vectorize(event_id0.__getitem__)(labels)  # convert labels into int

        events = np.concatenate((timestamps.reshape(-1, 1),
                                 np.zeros(timestamps.shape).astype(int).reshape(-1, 1),
                                 labels.reshape(-1, 1)), axis=1)
        # events visualizatiobn
        if SaveFig:
            color = {0: 'green', 100: 'red'}
            mne.viz.plot_events(events, raw_0.info['sfreq'], raw_0.first_samp, color=color,
                                event_id=event_id)

        # the difference between two full stimuli windows should be 7 sec.
        raw_0.n_times
        events = events[events[:, 2] == 0, :]  # keep only auditory stimuli
        stimt = np.append(events[:, 0], raw_0.n_times)  # stim interval
        epoch2keep = np.where(np.diff(stimt) == 3500)[0]  # keep only epoch of 7sec
        epoch2drop = np.where(np.diff(stimt) != 3500)[0]
        events = events[epoch2keep, :]

        print("subject " + subject + " data loaded")


        # ----------------------------
        # epoching
        # %%--------------------------
        if operations_to_apply["epoching"]:
            #            try:
            plt.close('all')
            # extract events from annotations
            event_id0 = {'BAD_data': 0, 'bad EPOCH': 100, 'BAD boundary': 100, 'EDGE boundary': 100}
            event_id1 = {'BAD_data': 1, 'bad EPOCH': 100, 'BAD boundary': 100, 'EDGE boundary': 100}
            tmp = zeta.data.stim.get_events(raw_0, event_id0)
            events0 = tmp[0][tmp[1], :]
            tmp = zeta.data.stim.get_events(raw_1, event_id1)
            events1 = tmp[0][tmp[1], :]

            # extract epochs
            event_id0, event_id1, tmin, tmax = {'low': 0}, {'high': 1}, 5.6, 7.6
            baseline = (None, 5.6)
            reject = dict(eeg=120)
            epochs0 = mne.Epochs(raw_0, events0, event_id0, tmin, tmax, baseline=baseline,
                                 reject=reject, preload=True, reject_by_annotation=0,
                                 verbose=verbose)

            epochs1 = mne.Epochs(raw_1, events1, event_id1, tmin, tmax, baseline=baseline,
                                 reject=reject, preload=True, reject_by_annotation=0,
                                 verbose=verbose)
            print("subject " + subject + " epoching done")

        ## ----------------------------
        # Classification 1
        # %%--------------------------

        epochs = mne.concatenate_epochs([epochs0, epochs1])
        if operations_to_apply["cla_ERPCovMDM"]:
            # %% 1
            PipelineTitle = 'cla_ERPCovMDM'
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
                'xdawn__nfilter': [8],
                # 'xdawn__bins':[[x for x in [0,20,40,60,80,100]]],
                # 'LASSO__cv': [inner_cv],  # static
                # 'LASSO__random_state': [42],  # static
                # 'LASSO__max_iter': [250]  # static
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
            # zeta.viz.classif.plot_cv_indices(outer_cv,X,y,)

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
            print("subject " + subject + " " + PipelineTitle + "done")

    print("Pipeline 4 DONE")
