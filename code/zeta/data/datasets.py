"""Zeta datasets"""

import mne
import numpy as np
import os
from itertools import compress
import zeta
import matplotlib.pyplot as plt


def get_dataset_list(data_folder=None):
    """loads the list of known datasets from a given folder
-
    Parameters
    ----------
    data_folder : str | None (default)
        path of the folder containing the datasets.

    Returns
    -------
    list_dataset_id : list
        a list containing the dataset_id in the given data_folder that are integrated in zeta.
        if data_folder is known, it will return all the datasets integrated in zeta module.
    """

    known_dataset_list=["raw_clean_32"    # High Versus Low inhibitory Stimuli of Tinnitus and control patients
                        ]
    if data_folder == None:
        dataset_list=known_dataset_list
    else:
        dataset_list=[]
        with os.scandir(data_folder) as it:
            for entry in it:
                if (not entry.name.startswith('.')
                        and not entry.is_file()
                        and entry.name in known_dataset_list) :
                    dataset_list.append(entry.name)

        if len(dataset_list) == 0:
            print("get_dataset_list: didn't found any compatible dataset")
    return dataset_list

def get_raw(data_folder, dataset_id, subject):
    """loads the subject data from a given dataset
-
    Parameters
    ----------
    data_folder : str
        path of the folder containing the dataset.
    dataset_id : str
        dataset_id code used in zeta module. Use get_dataset_list(data_folder) to find it.
    subject : str | int
        subject id (str) or its index (int) in the dataset.

    Returns
    -------
    list_dataset_id : list
        a list containing the dataset_id in the given data_folder that are integrated in zeta.
        if data_folder is known, it will return all the datasets integrated in zeta module.
    """

    if dataset_id=="raw_clean_32":
        return None


def get_dataset_low_v_high(data_folder, dataset_id, subject, ShowFig=False):
    """ Load Low Versus High Auditory Stimuli on Tinnitus patient and control.

    Note here that the EEG data are in µV while MNE use V. Therefore scale
    is with a 1e6 factor and it could cause a problem for non-linear related MNE
    analysing. I advice to apply a 1e-6 factor in the future to make sure that
    everything is working fine with mne.
    For classification, it is adviced to keep data in µV.
    """

    # clean loop variable
    runs = []
    labels = []
    events = []

    # load raw data
    for root, dirs, files in os.walk(os.path.join(data_folder, dataset_id)):
        for file in files:
            if file.startswith(subject):
                filepath = os.path.join(root, file)
                runs.append(mne.io.read_raw_fif(filepath, verbose="ERROR"))  # load data
                # events=mne.read_events(filepath)
                labels.append(file.split('_')[1])

    runs_0 = list(compress(runs, [x == 'low' for x in labels]))
    runs_1 = list(compress(runs, [x == 'high' for x in labels]))

    raw_0 = mne.concatenate_raws(runs_0)
    raw_1 = mne.concatenate_raws(runs_1)

    # rename table for event to annotations
    event_id0 = {'BAD_data': 0, 'bad EPOCH': 100, 'BAD boundary': 100, 'EDGE boundary': 100}

    # vizualization
    if ShowFig:
        scalings = dict(eeg=10e1)
        mne.viz.plot_raw(raw_0, scalings=scalings)

    timestamps = np.round(raw_0._annotations.onset * raw_0.info['sfreq']).astype(int)

    # raw_0._annotations.duration
    event_id = {'BAD_data': 0, 'bad EPOCH': 100}

    labels = raw_0._annotations.description
    labels = np.vectorize(event_id0.__getitem__)(labels)  # convert labels into int

    events = np.concatenate((timestamps.reshape(-1, 1),
                             np.zeros(timestamps.shape).astype(int).reshape(-1, 1),
                             labels.reshape(-1, 1)), axis=1)


    # extract events from annotations
    event_id0 = {'BAD_data': 0, 'bad EPOCH': 100, 'BAD boundary': 100, 'EDGE boundary': 100}
    event_id1 = {'BAD_data': 1, 'bad EPOCH': 100, 'BAD boundary': 100, 'EDGE boundary': 100}
    # tmp[0],tmp[1], tmp[2]: events, epochs2keep, epochs2drop
    tmp = zeta.data.stim.get_events(raw_0, event_id0)
    events0 = tmp[0][tmp[1], :]
    tmp = zeta.data.stim.get_events(raw_1, event_id1)
    events1 = tmp[0][tmp[1], :]

    # events visualizatiobn
    if ShowFig:
        fig,ax=plt.subplot(211)
        color = {0: 'green', 100: 'red'}
        mne.viz.plot_events(events, raw_0.info['sfreq'], raw_0.first_samp, color=color,
                            event_id=event_id)

    return raw_0, raw_1, events0, events1

def load_sessions_raw(data_folder, dataset_id, subject):
    """ Load all session of a subject

    Parameters
    ----------
    data_folder : str
        path of the folder containing the dataset.
    dataset_id : str
        dataset_id code used in zeta module. Use get_dataset_list(data_folder) to find it.
    subject : str | int
        subject id (str) or its index (int) in the dataset.

    Returns
    -------
    runs : list
        a list containing the raw file of the subject
    labels : list
        a list of session names.
    """
    runs = []
    labels = []
    sessions_path = []
    bad_sessions_path = []

    # the following verification will be required for handling several datasets
    # if dataset_id in ["raw_clean_32"]:

    # stack all subject's sessions
    for root, dirs, files in os.walk(os.path.join(data_folder, dataset_id)):
        for file in files:
            if file.startswith(subject):
                filepath = os.path.join(root, file)
                try:
                    runs.append(mne.io.read_raw_fif(filepath, verbose="ERROR"))  # stacks raw
                    # events=mne.read_events(filepath)  # might be usefull for some dataset

                    # WARNING: hardcoded label position could be troublesome in some dataset, check carefully
                    labels.append(file.split('_')[1])   # stacks session name
                    sessions_path.append(filepath)
                    subject_data_found=True
                except:
                    print("Couldn't load subject " + subject + " session " + file.split('_')[1] + " at " + filepath)
                    bad_sessions_path.append(filepath)
    if len(runs) == 0:
        print("Couldn't load any session of subject " + subject + " in dataset " + dataset_id)
    return runs, labels, sessions_path, bad_sessions_path