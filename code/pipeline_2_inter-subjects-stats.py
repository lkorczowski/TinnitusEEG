# -*- coding: utf-8 -*-
"""Perform group-level statistics based on subjects features

Parameters
----------
variable_name : variable_type
    variable_description


Returns
----------
variable_name : variable_type
    variable_description

See Also
--------
pipeline_1_loading_vizu

References
----------
[1]
    
:Authors:
    Louis Korczowski <louis.korczowski@gmail.com>

:History:
    | v0.4 2019-09-02 Added pandas sub-group requests
    | v0.3 2019-08-30 Group-Level Stats
    | v0.2 2019-08-29 .pickle importation from pipeline_1
    | v0.11 2019-08-28 removed omission frontiers module integration (see pipeline_3)
    | v0.1 2019-07-27 group-level integration with omission module

PROPERTY OF Zeta Technologies
CONFIDENTIAL
"""
# execution section
if __name__ == '__main__':
    ##==============================================================================
    # IMPORTS 
    # %%============================================================================

    import os, sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)

    # import matplotlib.pyplot as plt
    import mne
    import numpy as np
    import socket
    from itertools import compress
    import matplotlib.pyplot as plt
    import pickle
    import pandas as pd
    import zeta

    ##==============================================================================
    # PATHS & CONFIG
    # %%============================================================================
    """" CONFIGURE PATHS 
    This section will be integrate into zeta module with a use of 
    config file prepared for each computer."""
    # TODO: integrate this section into a module or a yalm file
    # WARNING: need to be the exact same than Pipeline_1 into we have a function to do it

    resultsID = 'pipeline_1_test'  # set ID for output directory (will replace any former results with same ID)
    ForceLoad = True  # TODO: need to be replaced with function checking if features are on hard drive
    SaveFig = False  # if set True, will overwrite previous figure in folder  resultsID

    # operation_to_apply should be the same list than in pipeline_2
    operations_to_apply = dict(
        epoching=1,
        GFP=0,  # Global Field Power
        TFR=0,  # Time-Frequency Response for each epoch
        TFR_av=1,  # Time-Frequency Response Averaging
        TFR_stats=1  # Compute inter-trials statistics on TFR
    )
    verbose = 'ERROR'
    subject = 1
    patient = 2  # patient group (static for a given dataset)
    session = 9  # 6 = 1 old replace (session 'high')
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
    zeta.util.mkdir(fig_dir)  # create results directory if needed

    """
    subjects=['patientPC', '5CA09', '05AY22', 'patient', '05GS16', '04MM25', 
          '1TG01', '05RP24', '3QO03', '5CD05', '5DN04', '05IN17', '05VP19', 
          'GS', '05MP21', '5DL08', '05BY20', '05DF18', '05MV11', '5NA09',
          'CQ', '5BB03', '05FV18', '5BY10', '04LK03', '04LM02', '05RM12',
          '2SN14', 'QE', '3NT07', '5GF07']  #obsolete, generated below
    # please note that 'patient_CR' is NOT a good name as the sparse with '_' is not
    # compatible with the other files. I recommand renamed it to 'patientCR' or 'CR'
    """

    ##==============================================================================
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

    ##==============================================================================
    # PROCESSING LOOP 
    # %%============================================================================
    tfr_av0 = []
    tfr_av1 = []
    tfr0_data = []
    tfr1_data = []
    subjects_list = []
    bad_subjects = []
    # TODO: move that loop for each processing ?
    for subject in list(subjects):  # [list(subjects)[1]]:  #
        ##----------------------------
        # RAW DATA LOADING AND PREPROCESSING
        # %%--------------------------
        fig_dir_sub = fig_dir + subject + os.path.sep
        zeta.utils.mkdir(fig_dir_sub)  # create results directory if needed

        # load subject data
        """
        note here that the EEG data are in µV while MNE use V. Therefore scale 
        is with a 1e6 factor andit could cause a problem for non-linear related MNE
        analysing. I advice to apply a 1e-6 factor in the future to make sure that
        everything is working fine with mne.
        For classification, it is adviced to keep data in µV.
        """

        ##----------------------------
        # Averaging of TFR (group-level)
        # %%--------------------------

        if operations_to_apply["TFR_av"]:
            print("Compute TFR_av")
            try:
                isNan = False
                if ForceLoad:
                    tfr_av0.append(
                        pickle.load(open(os.path.join(fig_dir_sub, "tfr_av0.pickle"), "rb"))
                    )
                    tfr0_data.append(tfr_av0[-1].data)
                    isNan = np.isnan(tfr0_data[-1]).any()
                    print("subject " + subject + " TFR_av0 loaded")

                if ForceLoad:
                    tfr_av1.append(
                        pickle.load(open(os.path.join(fig_dir_sub, "tfr_av1.pickle"), "rb"))
                    )
                    tfr1_data.append(tfr_av1[-1].data)
                    isNan = isNan or np.isnan(tfr1_data[-1]).any()
                    print("subject " + subject + " TFR_av1 loaded")

                subjects_list.append(subject)  # add only loaded data
                bad_subjects.append(isNan)

            except:
                print("subject " + subject + " TFR_av COULDN'T be loaded")
        else:
            print("Skip TFR_av")

    # ----------------------------
    # stats of TFR (group-level)
    # %%--------------------------
    if operations_to_apply["TFR_stats"]:
        print("Compute TFR_stats")

        #            try:

        # prepare data structures
        times = tfr_av0[0].times
        freqs = tfr_av0[0].freqs
        tfr0_data = np.asarray(tfr0_data)
        tfr1_data = np.asarray(tfr1_data)

        tests = [[0], [1], [2], [1, 2], [-1, 0, 1, 2]]
        dfs = pd.DataFrame(subjects_list)
        type_inhib = pd.read_csv(data_dir + os.path.sep + "type_inhib.csv")  # be sure to have the correct file here
        for group in tests:
            test_name = ''.join(str(e) for e in group)
            test_dir = fig_dir + 'group_' + test_name + os.path.sep
            # find subjects in test group
            group_names = type_inhib.loc[type_inhib['type_inhib'].isin(group)]['patient'].tolist()

            # get corresponding indexes
            keep = dfs.isin(group_names)[0].tolist()

            # keep subjects that respect different criteria
            subjects_to_keep = np.logical_and(~np.array(bad_subjects), np.array(keep))
            nb_sub = np.count_nonzero(subjects_to_keep == True)

            if nb_sub < 5:
                print("SKIP stats for group (%s), not enough subjects (%i/%i))" % (test_name, nb_sub, len(keep)))
            else:
                print("Perform stats for group (%s), included (%i/%i))" % (test_name, nb_sub, len(keep)))
                for ch_name in ['t7']:  # tfr_av0[0].info['ch_names']:  #
                    plt.close("all")
                    ch_ind = mne.pick_channels(tfr_av0[0].info['ch_names'], [ch_name])

                    # compute TF chuncks electrode by electrode
                    epochs_power_0 = tfr0_data[subjects_to_keep, ch_ind, :, :]  # only ch_name channel and remove NANs
                    epochs_power_1 = tfr1_data[subjects_to_keep, ch_ind, :, :]  # only ch_name channel and remove NANs

                    # compute permutation test (cluster-based)
                    """
                    About cluster-based statistics parameters:
                    A discussion and comparison of the threshold is available at the following:
                    https://www.nmr.mgh.harvard.edu/mne/stable/auto_tutorials/discussions/plot_background_statistics.html
                    The problem with hard-thresholding (e.g. threshold=6.0) is that it can either be very conservative
                    or venturesome depending of the target distribution. Therefore, using adaptative methods provided
                    by mne such as TFCE makes the cluster test more generalizable at the cost of complexity and 
                    computation.
                    """
                    threshold = 6.0
                    T_obs, clusters, cluster_p_values, H0 = \
                        mne.stats.permutation_cluster_test(
                            [epochs_power_0, epochs_power_1],
                            n_permutations=250, threshold=threshold, tail=0)

                    plt.figure()

                    # Create new stats image with only significant clusters
                    T_obs_plot = np.nan * np.ones_like(T_obs)
                    for c, p_val in zip(clusters, cluster_p_values):
                        if p_val <= 0.05:
                            T_obs_plot[c] = T_obs[c]

                    plt.imshow(T_obs,
                               extent=[times[0], times[-1], freqs[0], freqs[-1]],
                               aspect='auto', origin='lower', cmap='gray')
                    plt.imshow(T_obs_plot,
                               extent=[times[0], times[-1], freqs[0], freqs[-1]],
                               aspect='auto', origin='lower', cmap='RdBu_r')

                    plt.xlabel('Time (ms)')
                    plt.ylabel('Frequency (Hz)')
                    plt.title('Induced power (%s) type:%s, N=%i/%i' % (ch_name, test_name, nb_sub, len(keep)))

                    #                    plt.show()
                    plt.savefig(fname=test_dir + 'Pipeline_2_group-level_TF_cluster_stats_' + ch_name + '.png')
                print("Group " + test_name + " TFR_stats done")

    else:
        print("Skip TFR_stats")

    print("Pipeline 2 DONE")
