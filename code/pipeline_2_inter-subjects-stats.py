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

last version: DRAFT v0.5 (2019-07-27)
history:
    | v0.11 2019-08-28 removed omission frontiers module integration (see pipeline_3)
    | v0.1 2019-07-27 group-level integration with omission module

"""
#execution section
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
    
    #==============================================================================
    # PATHS & CONFIG
    #%%============================================================================
    """" CONFIGURE PATHS 
    This section will be integrate into zeta module with a use of 
    config file prepared for each computer."""
    # TODO: integrate this section into a module or a yalm file
    # WARNING: need to be the exact same than Pipeline_1 into we have a function to do it
    
    
    resultsID='pipeline_1_test' #set ID for output directory (will remplace any former results with same ID)
    ForceLoad=True # TODo: need to be remplaced with function checking if features are on hardrive
    SaveFig=False #if set True, will overwrite previous figure in folder  resultsID
    
    # operation_to_apply should be the same list than in pipeline_2
    operations_to_apply=dict(
            epoching=1,
            GFP=0, #Global Field Power
            TFR=0, #Time-Frequency Response for each epoch
            TFR_av=1, #Time-Frequency Response Averaging
            TFR_stats=0 #Compute inter-trials statistics on TFR
            )
    verbose='ERROR'
    subject=1
    patient =2 #patient group (static for a given dataset)
    session =9 #6 = 1 old remplacer apres (session 'high')
    ses2=8  # (session 'low')
    datasetname="raw_clean_32"
    
    #for automatic folder path use, add the elif: for your machine ID below
    configID=socket.gethostname()
    if configID=='your_machine_name':
        print("not configured")
    elif configID=='Crimson-Box':
         os.chdir("F:\\git\\TinnitusEEG\\code")
         data_dir = os.path.join("F:\\","data",'Zeta')
         fig_dir = os.path.join("D:\\", "GoogleDrive","Zeta Technologies","Zeta_shared","results")
    elif configID=='MacBook-Pro-de-Louis.local':
         os.chdir("/Volumes/Ext/git/TinnitusEEG/code")
         data_dir = os.path.join("/Volumes/Ext/","data",'Zeta')
         fig_dir='/Users/louis/Google Drive/Zeta Technologies/Zeta_shared/results'
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
    
    #==============================================================================
    # META DATA LOADING 
    #%%============================================================================  
    
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
    
    #==============================================================================
    # PROCESSING LOOP 
    #%%============================================================================  
    for subject in list(subjects): # [list(subjects)[1]]: #
        #----------------------------
        # RAW DATA LOADING AND PREPROCESSING
        #%%--------------------------
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

        #----------------------------
        # Averaging of TFR (group-level)
        #%%--------------------------
        tfr_av0=[]
        tfr_av1=[]
        
        if operations_to_apply["TFR_av"]:
            try:
                if ForceLoad:
                    tfr_av0.append(
                            pickle.load(open(os.path.join(fig_dir_sub,"tfr_av0.pickle"),"rb")).data
                        )
                if ForceLoad:
                    tfr_av1.append(
                            pickle.load(open(os.path.join(fig_dir_sub,"tfr_av1.pickle"),"rb")).data
                        )
                print("subject "+subject+" TFR_av loaded")
                
            except:
                print("subject "+subject+" TFR_av could be loaded")

            
            

    print("Pipeline 2 DONE")

