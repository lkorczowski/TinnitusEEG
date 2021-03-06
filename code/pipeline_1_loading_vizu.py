# -*- coding: utf-8 -*-
"""
Vizualize data and performance event-related statistics

:Authors:
    Louis Korczowski <louis.korczowski@gmail.com>

:history:

    | V1.2 2019-09-23 Refactored to integrate util and configuration modules
    | v1.1 2019-09-2O added datasets module integration (finally!)
    | v1.0 2019-09-18 added CSP
    | v0.71 2019-08-21 backup v0.5, pickle integration
    | v0.7 2019-07-27 group-level integration (CANCELED)
    | v0.6 2019-07-23 dataset integration (CANCELED)
    | v0.5 2019-07-19 correction of timing
    | v0.4 2019-07-17 TF statistical test and GFP
    | v0.3 2019-07-16 epoch extraction and TF vizualization
    | v0.2 2019-07-15 annotation to events convertion
    | v0.1 2019-07-11 pipeline creation

PROPERTY OF Robin Guillard EIRL
Provided only for public research purpose. Please contact the authors if you want to reuse that code.
"""
#execution section
import pandas as pd
import sklearn
from mpl_toolkits.axes_grid1 import make_axes_locatable

if __name__ == '__main__':
    #==============================================================================
    # IMPORTS 
    #%%============================================================================
    
    import os, sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    import zeta

    #import matplotlib.pyplot as plt
    import mne
    import numpy as np
    import socket
    from itertools import compress
    import matplotlib.pyplot as plt
    import pickle
    from scipy.signal import welch
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    #==============================================================================
    # PATHS & CONFIG
    #%%============================================================================
    """" CONFIGURE PATHS 
    This section will be integrate into zeta module with a use of 
    configuration.py file prepared for each computer."""
    # TODO: integrate this section into a module or a yalm file
    
    datasetname="raw_clean_32"

    resultsID='pipeline_1_test' #set ID for output directory (will remplace any former results with same ID)
    ForceSave=False #if set True, will overwrite previous results in Pickle
    SaveFig=False #if set True, will overwrite previous figure in folder  resultsID

    verbose='ERROR'

    data_dir, output_dir = zeta.configuration.load_directories()
    fig_dir=output_dir+os.path.sep+resultsID+os.path.sep   # pipeline output directory
    zeta.util.mkdir(fig_dir)                            # create results directory if needed

    # pipeline_1 specific configuration
    operations_to_apply=dict(
            epoching=1,
            GFP=0,       # Global Field Power
            TFR=0,       # Time-Frequency Response for each epoch
            TFR_av=0,    # Time-Frequency Response Averaging
            TFR_stats=0,  # Compute inter-trials statistics on TFR
            CSP=1
            )
    #==============================================================================
    # META DATA LOADING 
    #%%============================================================================  

    subjects=zeta.data.datasets.get_subjects_info(data_dir, datasetname).keys()

    #==============================================================================
    # PROCESSING LOOP 
    #%%============================================================================
    auc=[]
    subject_list=[]
    for subject in [list(subjects)[1]]: #list(subjects): #
        raw_0, raw_1, events0, events1=[], [], [], []
        try:
            #----------------------------
            # RAW DATA LOADING AND PREPROCESSING
            #%%--------------------------
            fig_dir_sub=fig_dir+subject+os.path.sep
            zeta.util.mkdir(fig_dir_sub)  # create results directory if needed

            #load raw data
            raw_0, raw_1, events0, events1 = zeta.data.datasets.get_raw(data_dir, datasetname, subject)

            #vizualization
            if SaveFig:
                scalings=dict(eeg=10e1)
                mne.viz.plot_raw(raw_0,scalings=scalings)

            # events visualizatiobn
            if SaveFig:
                event_id = {'BAD_data': 0, 'bad EPOCH': 100}
                color = {0: 'green', 100: 'red'}
                mne.viz.plot_events(events0, raw_0.info['sfreq'], raw_0.first_samp, color=color,
                                event_id=event_id)

            print("subject "+subject+" data loaded")

            #----------------------------
            # GLOBAL FIELD POWER (GFP)
            #%%--------------------------
            if operations_to_apply["GFP"]:
                #try:
                plt.close('all')
                reject=dict(eeg=120) # in µV
                """LOW VERSUS HIGH AUDITORY DATASET INFO
                stimulus code is -0.6 before auditory stimulus
                auditory stimulus last 5 sec
                post-stimulus resting state last 7sec
                therefore full epoch:
                tmin,tmax=0.6,7.6
                auditory stimuli:
                tmin,tmax=0.6,5.6
                post-stimuli:
                 tmin,tmax=5.6,7.6
                """
                event_id, tmin, tmax = 0, 5.1, 7.6 #post-stimuli -0.5 to 2 seconds
                baseline = (5.1,5.6) #post-stimuli -0.5 to 0 (during auditory stimuli)
                offset=raw_0.info['sfreq']*0.6
                iter_freqs = [
                    ('Theta', 4, 7),
                    ('Alpha', 8, 12),
                    ('Beta', 13, 25),
                    ('Gamma', 30, 45)
                ]

                #Perform GFP
                options=dict(iter_freqs=iter_freqs,baseline=baseline,
                       event_id=event_id,tmin=tmin, tmax=tmax,reject=reject,
                       verbose=verbose)

                tmp=zeta.data.stim.get_events(raw_0,event_id0,offset=0)
                events0=tmp[0][tmp[1],:] #keep only good data
                gfp0=zeta.analysis.freq.GFP(raw_0,events0,**options)
                if SaveFig:
                    gfp0[1].savefig(fig_dir_sub+'gfp0',dpi=300)

                tmp=zeta.data.stim.get_events(raw_1,event_id0)
                events1=tmp[0][tmp[1],:]
                gfp1=zeta.analysis.freq.GFP(raw_1,events1,**options)
                if SaveFig:
                    gfp1[1].savefig(fig_dir_sub+'gfp1',dpi=300)

                print("subject "+subject+" GFP done")



            #----------------------------
            # epoching
            #%%--------------------------
            if operations_to_apply["epoching"]:
    #            try:
                plt.close('all')
                #extract events from annotations
                event_id0={'BAD_data': 0, 'bad EPOCH': 100, 'BAD boundary': 100, 'EDGE boundary': 100}
                event_id1={'BAD_data': 1, 'bad EPOCH': 100, 'BAD boundary': 100, 'EDGE boundary': 100}
                tmp=zeta.data.stim.get_events(raw_0,event_id0)
                events0=tmp[0][tmp[1],:]
                tmp=zeta.data.stim.get_events(raw_1,event_id1)
                events1=tmp[0][tmp[1],:]

                #extract epochs
                event_id0,event_id1, tmin, tmax = {'low':0},{'high':1}, 0.6, 7.6
                baseline = (None,5.6)
                reject=dict(eeg=120)
                epochs0 = mne.Epochs(raw_0, events0, event_id0, tmin, tmax, baseline=baseline,
                                    reject=reject, preload=True,reject_by_annotation=0,
                                    verbose=verbose)

                epochs1 = mne.Epochs(raw_1, events1, event_id1, tmin, tmax, baseline=baseline,
                                reject=reject, preload=True,reject_by_annotation=0,
                                verbose=verbose)
                print("subject "+subject+" epoching done")


            #----------------------------
            # TIME-FREQUENCY ANALYSIS (TFR)
            #%%--------------------------
            if operations_to_apply["TFR"]:
                """Apply averaged Morlet periodogram estimation to return the instantaneous power spectral density.
                Epochs are averaged.
                Save figures ONLY
                """
    #            try:
                options=dict(iter_freqs=iter_freqs,baseline=baseline,
                             tmin=tmin, tmax=tmax,ch_name='cz')
                tfr0=zeta.analysis.freq.TF(epochs0,**options)
                tfr1=zeta.analysis.freq.TF(epochs1,**options)
                if SaveFig:
                    tfr0[2].savefig(fig_dir_sub+'TF1_low',dpi=300)
                    tfr0[3].savefig(fig_dir_sub+'TF2_low',dpi=300)
                    tfr0[4].savefig(fig_dir_sub+'TF3_low',dpi=300)
                    tfr1[2].savefig(fig_dir_sub+'TF1_high',dpi=300)
                    tfr1[3].savefig(fig_dir_sub+'TF2_high',dpi=300)
                    tfr1[4].savefig(fig_dir_sub+'TF3_high',dpi=300)
                print("subject "+subject+" TFR done")


            #----------------------------
            # Averaging of TFR
            #%%--------------------------
            decim=1
            freqs = np.arange(0.5, 40, 0.5)
            n_cycles = freqs / 2.

            if operations_to_apply["TFR_av"]:
                """Apply averaged Morlet periodogram estimation to return the instantaneous power spectral density.
                Epochs are averaged.
                Save Values ONLY
                """
    #            try:
                tfr_av0 = mne.time_frequency.tfr_morlet(epochs0, freqs,
                                          n_cycles=n_cycles, decim=decim,
                                          return_itc=False, average=True)
                if ForceSave:
                    with open(os.path.join(fig_dir_sub,"tfr_av0.pickle"),"wb") as f:
                            pickle.dump(tfr_av0, f)

                tfr_av1 = mne.time_frequency.tfr_morlet(epochs1, freqs,
                                          n_cycles=n_cycles, decim=decim,
                                          return_itc=False, average=True)
                if ForceSave:
                    with open(os.path.join(fig_dir_sub,"tfr_av1.pickle"),"wb") as f:
                            pickle.dump(tfr_av1, f)
                print("subject "+subject+" TFR_av done")



            #----------------------------
            # CLUSTER-BASED PERMUTATION BASED ON TFR
            #%%--------------------------
            if operations_to_apply["TFR_stats"]:
    #            try:
                plt.close('all')
                for ch_name in ['t7']:#epochs0.info['ch_names']:#

                    #compute TF chuncks electrode by electrode
                    epochs0c=epochs0.copy().pick_channels([ch_name])
                    epochs1c=epochs1.copy().pick_channels([ch_name])
                    tfr_epochs_0 = mne.time_frequency.tfr_morlet(epochs0c, freqs,
                                              n_cycles=n_cycles, decim=decim,
                                              return_itc=False, average=False)

                    tfr_epochs_1 = mne.time_frequency.tfr_morlet(epochs1c, freqs,
                                              n_cycles=n_cycles, decim=decim,
                                              return_itc=False, average=False)



                    epochs_power_0 = tfr_epochs_0.data[:, 0, :, :]  # only ch_name channel as 3D matrix
                    epochs_power_1 = tfr_epochs_1.data[:, 0, :, :]  # only ch_name channel as 3D matrix

                    #compute permutation test (cluster-based)
                    threshold = None
                    T_obs, clusters, cluster_p_values, H0 = \
                    mne.stats.permutation_cluster_test([epochs_power_0, epochs_power_1],
                                                       n_permutations=250, threshold=threshold, tail=0)

                    times = 1e3 * epochs0.times  # change unit to ms

                    evoked_condition_0 = epochs0c.average()
                    evoked_condition_1 = epochs1c.average()

                    plt.figure()
                    plt.subplots_adjust(0.12, 0.08, 0.96, 0.94, 0.2, 0.43)

                    plt.subplot(2, 1, 1)
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
                    plt.title('Induced power (%s)' % ch_name)

                    ax2 = plt.subplot(2, 1, 2)
                    evoked_contrast = mne.combine_evoked([evoked_condition_0,
                                                          evoked_condition_1],
                                                          weights=[1, -1])
                    if SaveFig:
                        evoked_contrast.plot(axes=ax2, time_unit='s')
                        plt.show()
                        plt.savefig(fname=fig_dir_sub+ 'E_01_TF_cluster_stats_'+ ch_name + '.png')
                print("subject "+subject+" TFR_stats done")

            ## ----------------------------
            # Quick Classification and visualization of the features
            # %%--------------------------
            if operations_to_apply["CSP"]:
                """Computes the Common Spatial Pattern (CSP) on all data and print the most discriminant feature and 
                performs a simple CSP+Logistic Regression Classification. The results will be stacked for all subjects
                at the end of pipeline_1.
                code inspired by Alexandre Barachant's Kaggle
                https://www.kaggle.com/alexandrebarachant/common-spatial-pattern-with-mne
                another reading for CSP decoding
                https://www.nmr.mgh.harvard.edu/mne/dev/auto_examples/decoding/plot_decoding_csp_eeg.html
                """

                plt.close('all')
                # extract events from annotations
                event_id0 = {'BAD_data': 0, 'bad EPOCH': 100, 'BAD boundary': 100, 'EDGE boundary': 100}
                event_id1 = {'BAD_data': 1, 'bad EPOCH': 100, 'BAD boundary': 100, 'EDGE boundary': 100}
                # tmp[0],tmp[1], tmp[2]: events, epochs2keep, epochs2drop
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

                epochs = mne.concatenate_epochs([epochs0, epochs1])
                # get data
                X = epochs.get_data()
                tmp = epochs.copy().drop(epochs.events[:, 2] < 0)  # remove bad epochs
                y = tmp.events[:, 2]
                zeta.stats.classif.spatio_spectral_patterns(epochs, y, n_components=4, output_dir="", test_name="test")
        except:
            print('could load and process the subject '+ subject)

    if operations_to_apply["CSP"]:
        auc = pd.DataFrame(data=auc, index=subject_list, columns=['auc'])
        auc.to_csv(fig_dir+'cross_val_auc.csv')
        plt.figure(figsize=(4, 4))
        auc.plot(kind='bar', y='auc')
        plt.xlabel('Subject')
        plt.ylabel('AUC')
        plt.title('During Vs. After classification')
        plt.savefig(fig_dir+'cross_val_auc.png', bbox_inches='tight')
        print("CSP: Inter-subject results saved for %i / %i subjects (couldn't compute for the others)" % (len(auc), len(subjects)))


    print("Pipeline 1 DONE")


