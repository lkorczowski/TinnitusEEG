"""Zeta stimulis and annotation convertion"""
import numpy as np


def get_events(raw,event_id,offset=0):
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
    offset : int (default=0)
        Apply an offset in sample to all events. Example:
            offset=-60 will shift all events to 60 samples before the annotation

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
    # extract time stamps from annotations
    timestamps = np.round(raw._annotations.onset*raw.info['sfreq']+offset).astype(int)
    assert np.all(timestamps < raw.n_times), "offset overflow total data length"

    # get labels
    labels = raw._annotations.description
    labels = np.vectorize(event_id.__getitem__)(labels) #convert labels into int
    
    # build event matrix
    events = np.concatenate((timestamps.reshape(-1,1),
                               np.zeros(timestamps.shape).astype(int).reshape(-1,1),
                               labels.reshape(-1,1)),axis=1)
        
    # the difference between two full stimuli windows should be 7 sec. 
    events = events[events[:, 2] < 100, :] #keep only events and remove annotations

    assert np.unique(events[:, 2]).size ==1 #TODO: make it works for different events
    
    stimt = np.append(events[:, 0], raw.n_times) #stim interval
    epochs2keep = np.where(np.diff(stimt) == raw.info['sfreq']*7)[0] #TODO: keep only epoch of 7sec (make it an argument)
    epochs2drop = np.where(np.diff(stimt) != raw.info['sfreq']*7)[0] #drop the rest

    return events, epochs2keep, epochs2drop

def map_targets(y, mapping=None):
    """ Get events from specific Zeta dataset.
            Events are extracted from annotations.
            example:
                import zeta
                tmp=zeta.data.stim.get_events(raw,event_id)
                events=tmp[0][tmp[1],:]#extract only interesting events

        Parameters
        ----------
        y : ndarray, list or ndarray
            the initial target
        mapping : dict, ndarray or None (default)
            the map to convert target. If None, it will just return y.


        Returns
        ----------
        y_converted : ndarray, list or ndarray (same as y)
            the converted target

        See Also
        --------

        References
        ----------
        [1]

        """
    y_converted = []

    if mapping is None:
        y_converted = y
    else:
        if isinstance(mapping, list) or isinstance(mapping, (np.ndarray, np.generic)):
            if isinstance(y[0], list) or isinstance(y[0], (np.ndarray)): # if nested targets
                y_converted = y.copy()

                print("array of array1")

                for indy, y_tmp in enumerate(y):
                    y_converted[indy] = mapping[y_tmp]
            else: # if list
                print("array1")

                y_converted = np.array(mapping[y])

        elif isinstance(mapping, dict):
            if isinstance(y[0], list) or isinstance(y[0], (np.ndarray)): # if nested targets
                y_converted = y.copy()

                print("array of array2")
                for indy, y_tmp in enumerate(y):
                    y_converted[indy] = [mapping.get(y_tmp2) for y_tmp2 in y_tmp]
            else:
                print("array2")

                y_converted = np.array([mapping.get(y_tmp) for y_tmp in y])
        else:
            raise TypeError('y must be list, ndarray, dict or None')

    return y_converted

if __name__ == '__main__':
    # unit test map_targets()
    y1 = np.array([1, 2, 3, 4])
    y2 = np.array([3, 2, 1, 4])
    y3 = np.array([0, 2, 1, 4, 5])

    mapping = {1: 0, 2: 0, 3: 1, 4: 1}

    y = [y1, y2, y3]

    y_conv = map_targets(y, mapping=mapping)
    print("TEST map_targets() WITH DICT")
    print("y")
    print(y)
    print("converted y1")
    print(map_targets(y1, mapping=mapping))
    print("converted y")
    print(map_targets(y, mapping=mapping))

    mapping = np.array([None, 0, 0, 1, 1, None])

    print("TEST map_targets() WITH ARRAY")
    print("converted y1")
    print(map_targets(y1, mapping))
    print("converted y")
    print(map_targets(y, mapping))


    print(map_targets(y, mapping=None))
