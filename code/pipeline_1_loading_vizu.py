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
    | v0.5 2019-07-19 correction of timing
    | v0.4 2019-07-17 TF statistical test and GFP
    | v0.3 2019-07-16 epoch extraction and TF vizualization
    | v0.2 2019-07-15 annotation to events convertion
    | v0.1 2019-07-11 pipeline creation

"""
#execution section
if __name__ == '__main__':
    import os, sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    import zeta

    #import matplotlib.pyplot as plt
    import mne
    import numpy as np
    import socket
    from pathlib import Path #should be used instead of os.path.join
    from itertools import compress
    import matplotlib.pyplot as plt
    #%%
    """" CONFIGURE PATHS 
    This section will be integrate into zeta module with a use of 
    config file prepared for each computer."""
    resultsID='pipeline_1' #set ID for output directory (will remplace any former results with same ID)
        
    #for automatic folder path use, add the elif: for your machine ID below
    configID=socket.gethostname()
    if configID=='your_machine_name':
        print("not configured")
    elif configID=='Crimson-Box':
         os.chdir("F:\\git\\TinnitusEEG\\code")
         data_dir = os.path.join("F:\\","data",'Zeta') #gitlab "freiburg" directory
         fig_dir = os.path.join("D:\\", "GoogleDrive","Zeta Technologies","Zeta_shared","results")
    elif configID=='MacBook-Pro-de-Louis.local':
         os.chdir("/Volumes/Ext/git/TinnitusEEG/code")
         data_dir = os.path.join("/Volumes/Ext/","data",'Zeta') #gitlab "freiburg" directory
         fig_dir = "/Volumes/Ext/python/zeta/results/"
    else:
        print('config not recognize, please add the path of your git directories')
    #WARNING : "\\" is used for windows, "/" is used for unix (or 'os.path.sep')

    fig_dir=fig_dir+os.path.sep+resultsID+os.path.sep #pipeline output directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir) #create results directory if needed
    
    """
    subjects=['patientPC', '5CA09', '05AY22', 'patient', '05GS16', '04MM25', 
          '1TG01', '05RP24', '3QO03', '5CD05', '5DN04', '05IN17', '05VP19', 
          'GS', '05MP21', '5DL08', '05BY20', '05DF18', '05MV11', '5NA09',
          'CQ', '5BB03', '05FV18', '5BY10', '04LK03', '04LM02', '05RM12',
          '2SN14', 'QE', '3NT07', '5GF07']  #obsolete, generated below
    # please note that 'patient_CR' is NOT a good name as the sparse with '_' is not
    # compatible with the other files. I recommand renamed it to 'patientCR' or 'CR'
    """
    
    # %%
    """ DATA LOADING 
    this section will load data and extract the epochs with conditions.
    """
    subject=1
    patient =2 #patient group (static for a given dataset)
    session =9 #6 = 1 old remplacer apres (session 'high')
    ses2=8  # (session 'low')
    datasetname="raw_clean_32"
    ShowFigures=1
    
    names = os.listdir(os.path.join(data_dir, datasetname, str(patient)+ "_"+ str(session)))
    names2 = os.listdir(os.path.join(data_dir, datasetname, str(patient)+ "_"+ str(ses2)))
    
    pat=[]
    pat2=[]
    for name in names:
        #print name.split('_')[0]
        pat.append(name.split('_')[0]) #all subjects ID from names
    for name in names2:
        #print name.split('_')[0]
        pat2.append(name.split('_')[0]) #all subjects ID from names2
        
    cong={} #build of dictionnary of all session for each subject
    for name in names2:
            if pat.__contains__(name.split('_')[0]):
                if cong.keys().__contains__(name.split('_')[0]):
                    cong[name.split('_')[0]].append(name) #add file to the list
                else:
                    cong[name.split('_')[0]]=[name] #add first file to the list
    for name in names:
            if pat2.__contains__(name.split('_')[0]):
                cong[name.split('_')[0]].append(name)
    
    t=["exacerb","absente","partielle","totale"]
    subjects=cong.keys()
    
    
    for subject in [list(subjects)[1]]: # subject in list(subjects): # 
        fig_dir_sub=fig_dir+subject+os.path.sep
        if not os.path.exists(fig_dir_sub):
            os.makedirs(fig_dir_sub) #create results directory if needed
        #load subject data
        """
        note here that the EEG data are in µV while MNE use V. Therefore scale 
        is with a 1e6 factor andit could cause a problem for non-linear related MNE
        analysing. I advice to apply a 1e-6 factor in the future to make sure that
        everything is working fine with mne.
        For classification, it is adviced to keep data in µV.
        """
        runs=[]
        labels=[]
        events=[]
        for root, dirs, files in os.walk(os.path.join(data_dir,datasetname)):
            for file in files:
                if file.startswith(subject):
                    filepath=os.path.join(root,file)
                    runs.append(mne.io.read_raw_fif(filepath)) #load data
                    #events=mne.read_events(filepath)
                    labels.append(file.split('_')[1])
                    
        eventscode=dict(zip(np.unique(runs[0]._annotations.description),[0,1]))
        #events=mne.events_from_annotations(runs[0])       
                
        runs_0=list(compress(runs,[x=='low' for x in labels]))
        runs_1=list(compress(runs,[x=='high' for x in labels]))
        
        raw_0=mne.concatenate_raws(runs_0)
        raw_1=mne.concatenate_raws(runs_1)
    
        #rename table for event to annotations
        event_id0 = {'BAD_data': 0, 'bad EPOCH': 100, 'BAD boundary':100,'EDGE boundary':100}
        
        #vizualization
        if ShowFigures:
            scalings=dict(eeg=10e1)
            mne.viz.plot_raw(raw_0,scalings=scalings)
        
        timestamps=np.round(raw_0._annotations.onset*raw_0.info['sfreq']).astype(int)
        
        raw_0._annotations.duration
        event_id = {'BAD_data': 0, 'bad EPOCH': 100}
        
        labels=raw_0._annotations.description
        labels=np.vectorize(event_id0.__getitem__)(labels) #convert labels into int
        
        events=np.concatenate((timestamps.reshape(-1,1),
                                   np.zeros(timestamps.shape).astype(int).reshape(-1,1),
                                   labels.reshape(-1,1)),axis=1)
        # events visualation
        if ShowFigures:
            color = {0: 'green', 100: 'red'}
            mne.viz.plot_events(events, raw_0.info['sfreq'], raw_0.first_samp, color=color,
                            event_id=event_id)
            
        # the difference between two full stimuli windows should be 7 sec. 
        raw_0.n_times
        events=events[events[:,2]==0,:] #keep only auditory stimuli
        stimt=np.append(events[:,0],raw_0.n_times) #stim interval
        epoch2keep=np.where(np.diff(stimt)==3500)[0] #keep only epoch of 7sec
        epoch2drop=np.where(np.diff(stimt)!=3500)[0]
        events=events[epoch2keep,:]
        
        
        
        # %% GFP analysis
        plt.close('all')
        reject=dict(eeg=120)
        event_id, tmin, tmax = 0, 5.1, 7.6
        baseline = (5.1,5.6)
        offset=raw_0.info['sfreq']*0.6
        iter_freqs = [
            ('Theta', 4, 7),
            ('Alpha', 8, 12),
            ('Beta', 13, 25),
            ('Gamma', 30, 45)
        ]
        verbose='ERROR'
        tmp=zeta.data.stim.get_events(raw_0,event_id0,offset=0)
        events0=tmp[0][tmp[1],:] #keep only good data
        
        options=dict(iter_freqs=iter_freqs,baseline=baseline,
                               event_id=event_id,tmin=tmin, tmax=tmax,reject=reject,
                               verbose=verbose)
        out0=zeta.analysis.freq.GFP(raw_0,events0,**options)
        
        out0[1].savefig(fig_dir_sub+'gfp0',dpi=300)
        tmp=zeta.data.stim.get_events(raw_1,event_id0)
        events1=tmp[0][tmp[1],:]
        
        out1=zeta.analysis.freq.GFP(raw_1,events1,**options)
        out1[1].savefig(fig_dir_sub+'gfp1',dpi=300)
    
        
        # %% D Time-Frequency analysis
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
        
        options=dict(iter_freqs=iter_freqs,baseline=baseline,
                     tmin=tmin, tmax=tmax,ch_name='cz')
        out0=zeta.analysis.freq.TF(epochs0,**options)
        out1=zeta.analysis.freq.TF(epochs1,**options)
    
        out0[2].savefig(fig_dir_sub+'TF1_low',dpi=300)
        out0[3].savefig(fig_dir_sub+'TF2_low',dpi=300)
        out0[4].savefig(fig_dir_sub+'TF3_low',dpi=300)
        out1[2].savefig(fig_dir_sub+'TF1_high',dpi=300)
        out1[3].savefig(fig_dir_sub+'TF2_high',dpi=300)
        out1[4].savefig(fig_dir_sub+'TF3_high',dpi=300)
        # %% E Cluster-based TF statistics for each electrode
        #parameters
        plt.close('all')
        for ch_name in ['t7']:#epochs0.info['ch_names']:#
            decim=1
            freqs = np.arange(0.5, 40, 0.5) 
            n_cycles = freqs / 2.
            
            
            #compute TF chuncks
            epochs0c=epochs0.copy()
            epochs1c=epochs1.copy()
            tfr_epochs_0 = mne.time_frequency.tfr_morlet(epochs0c.pick_channels([ch_name]), freqs,
                                      n_cycles=n_cycles, decim=decim,
                                      return_itc=False, average=False)
            
            tfr_epochs_1 = mne.time_frequency.tfr_morlet(epochs1c.pick_channels([ch_name]), freqs,
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
            evoked_contrast = mne.combine_evoked([evoked_condition_0.pick_channels([ch_name]), 
                                                  evoked_condition_1.pick_channels([ch_name])],
                                                 weights=[1, -1])
            evoked_contrast.plot(axes=ax2, time_unit='s')
            
            plt.show()
            plt.savefig(fname=fig_dir_sub+ 'E_01_TF_cluster_stats_'+ ch_name + '.png')
            
            assert False
    
    
    
