"""Zeta stimulis and annotation convertion"""
import numpy as np

def get_events(raw,event_id):
    """ Get events from specific Zeta dataset. 
        Events are extracted from annotations.
        example:
            import zeta
            tmp=zeta.data.stim.get_events(raw,event_id)
            events=tmp[0][tmp[1],:]#extract only interesting events 
    
    Parameters
    ----------
    raw : mne.io.raw instance
        the raw data
    event_id : dict
          the dictionnary used for annotation selections
          only labels with int<100
          example: 
              # only annotation with 'BAD_data' are translated into events
              event_id = {'BAD_data': 0, 'bad EPOCH': 100, 'BAD boundary':100,'EDGE boundary':100}

    Returns
    ----------
    events : narray, shape (n_trials, 3)
        the events matrix used for mne.
    epochs2keep : narray
        the list of events respecting the condition for this specific dataset
    epochs2drop : narray
        the list of events not used for this specific dataset
        
    See Also
    --------
    
    References
    ----------
    [1]
    

    TODO: [] make condition epochs2keep as argument (length of the epoch)
    TODO: [] compatibility for different datasets
    
    """
    #extract time stamps from annotations
    timestamps=np.round(raw._annotations.onset*raw.info['sfreq']).astype(int)
    
    #get labels
    labels=raw._annotations.description
    labels=np.vectorize(event_id.__getitem__)(labels) #convert labels into int
    
    #build event matrix
    events=np.concatenate((timestamps.reshape(-1,1),
                               np.zeros(timestamps.shape).astype(int).reshape(-1,1),
                               labels.reshape(-1,1)),axis=1)
        
    # the difference between two full stimuli windows should be 7 sec. 
    events=events[events[:,2]<100,:] #keep only events and remove annotations
    
    assert np.unique(events[:,2]).size==1 #TODO: make it works for different events
    
    stimt=np.append(events[:,0],raw.n_times) #stim interval
    epochs2keep=np.where(np.diff(stimt)==raw.info['sfreq']*7)[0] #TODO: keep only epoch of 7sec (make it an argument)
    epochs2drop=np.where(np.diff(stimt)!=raw.info['sfreq']*7)[0] #drop the rest
    return events, epochs2keep, epochs2drop