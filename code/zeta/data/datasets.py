"""Zeta datasets
Methods consist
get_dataset_list()   # get compatible dataset names
get_subjects_info()  # return a dictionnary of subjects with their sessions
get_raw()            # return all sessions of a given subject for a given dataset

"""

import mne
import numpy as np
import os
from itertools import compress
import zeta
import matplotlib.pyplot as plt
import pandas as pd


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

    known_dataset_list = ["raw_clean_32",  # High Versus Low inhibitory Stimuli of Tinnitus and control patients
                          "Distress2010",  # Tinnitus Distress patients (1, 2, 3, 4 Distress) - resting state
                          "NormativeDB",   # Control Patient for Distress2010 - resting state (250+ files)
                          "Tinnitus_EEG"   # augmented dataset from Distress2010 (300+ subjects)
                          ]
    if data_folder is None:
        dataset_list = known_dataset_list
    else:
        dataset_list = []
        with os.scandir(data_folder) as it:
            for entry in it:
                if (not entry.name.startswith('.')
                        and not entry.is_file()
                        and entry.name in known_dataset_list):
                    dataset_list.append(entry.name)

        if len(dataset_list) == 0:
            print("get_dataset_list: didn't found any compatible dataset in folder " + data_folder)
    return dataset_list


def get_subjects_info(data_folder, dataset_id, format="dict"):
    """return a dictionary of subjects and associated files from known dataset from a given folder
-
    Parameters
    ----------
    data_folder : str
        path of the folder containing the datasets (can be checked using get_dataset_list).
    dataset_id : str
        dataset_id code used in zeta module. Use get_dataset_list(data_folder) to find it.
    format : str
        "dict" or "DataFrame". By default return a dict of subject containing a dict of sessions ("dict") but using
        "DataFrame" return has a DataFrame.

    Returns
    -------
    subjects_info : dict (default) | DataFrame
        a list dict the subjects_id and associated files. You can use dict_subjects_id.keys() to get the subjects_list.

    """
    subjects_info = {}  # build of dictionnary of all session for each subject

    if dataset_id == "raw_clean_32":
        """ High Versus Low inhibitory Stimuli of Tinnitus and control patients
        """
        patient = 2  # patient group (static for a given dataset)
        session = 9  # 6 = 1 old remplacer apres (session 'high')
        ses2 = 8  # (session 'low')
        names = os.listdir(os.path.join(data_folder, dataset_id, str(patient) + "_" + str(session)))
        names2 = os.listdir(os.path.join(data_folder, dataset_id, str(patient) + "_" + str(ses2)))

        pat = []
        pat2 = []
        for name in names:
            # print name.split('_')[0]
            pat.append(name.split('_')[0])  # all subjects ID from names
        for name in names2:
            # print name.split('_')[0]
            pat2.append(name.split('_')[0])  # all subjects ID from names2

        for name in names2:
            if pat.__contains__(name.split('_')[0]):
                if subjects_info.keys().__contains__(name.split('_')[0]):
                    subjects_info[name.split('_')[0]].append(name)  # add file to the list
                else:
                    subjects_info[name.split('_')[0]] = [name]  # add first file to the list
        for name in names:
            if pat2.__contains__(name.split('_')[0]):
                subjects_info[name.split('_')[0]].append(name)

    elif dataset_id == "Distress2010":
        """ High Versus Low Distress patients (1, 2, 3, 4 Distress)
        """
        sub_high = 'high distress'
        sub_low = 'low distress'
        filenames = os.listdir(os.path.join(data_folder, dataset_id, sub_high)) + \
                    os.listdir(os.path.join(data_folder, dataset_id, sub_low))

        # get all subjects ID
        valid_id = ["1", "2", "3", "4"]  # Distress group (file begin with)
        valid_info = ["#", "NBN", "LI", "RE", "ICON", "BIL", "PT", "128", "123", "128I", "128#", "HOLOCRANIAL", "HOLO",
                      "HOLOC", "HYPERAC", "CONCENTRATIESTN"]  # info to sparse from file name

        for filename in filenames:
            if filename[0] in valid_id:
                symptoms, subjectname = _sparse_info_from_file(filename.split(".")[0], valid_info, separator="_")
                symptoms.append({"distress": filename[0]})
                paradigm = "rest"
                session_info = {"paradigm": paradigm, "symptoms": symptoms}

                try:
                    subjects_info[subjectname].update(
                        {filename: session_info}  # add new session
                    )

                except KeyError:
                    subjects_info[subjectname] = {filename: session_info}  # create session`
    elif dataset_id == "NormativeDB":
        """ Control subjects in resting state
        """
        filenames = os.listdir(os.path.join(data_folder, dataset_id, "clean-up", "M")) + \
                    os.listdir(os.path.join(data_folder, dataset_id, "clean-up", "F"))

        # get all subjects ID
        valid_id = ["1", "2", "3", "4"]  # Distress group (file begin with)
        valid_info = ["#", "NBN", "LI", "RE", "ICON", "BIL", "PT", "128", "123", "128I", "128#", "HOLOCRANIAL", "HOLO",
                      "HOLOC", "HYPERAC", "CONCENTRATIESTN"]  # info to sparse from file name

        for filename in filenames:
            if filename[0] in valid_id:
                symptoms, subjectname = _sparse_info_from_file(filename.split(".")[0], valid_info, separator="_")
                symptoms.append("Control")
                symptoms.append({"distress": "0"})
                paradigm = "rest"
                session_info = {"paradigm": paradigm, "symptoms": symptoms, "gender": filename[2]}

                try:
                    subjects_info[subjectname].update(
                        {filename: session_info}  # add new session
                    )

                except KeyError:
                    subjects_info[subjectname] = {filename: session_info}  # create session

    else:
        print("get_subjects_info: unknown dataset")
    if format == "DataFrame":
        subjects_info
    return subjects_info


def get_raw(data_folder, dataset_id, subject):
    """loads the subject data (raw and events) from a given dataset and organize it into several conditions if available

    raw_0, raw_1, events0, events1 = get_raw(data_folder, dataset_id, subject)
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

    if dataset_id in ["raw_clean_32"]: #two conditions experiments
        raw_0, raw_1, events0, events1 = get_dataset_low_v_high(data_folder, dataset_id, subject, ShowFig=False)
    if dataset_id in ["Distress2010","NormativeDB"]:
        raw_0, raw_1, events0, events1 = get_dataset_distress(data_folder, dataset_id, subject, ShowFig=False)

    else:
        print(dataset_id + ": Unknown dataset, please check compatible dataset using get_dataset_list().")
        print(dataset_id + ": if you want, you can add this dataset to get_dataset_list() and get_raw(),")
        print(dataset_id + ": OR you can use directly load_sessions_raw() if your data respect a known structure.")

    return raw_0, raw_1, events0, events1


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

    runs, labels, _, _ = load_sessions_raw(data_folder, dataset_id, subject)  # load all session from this subject

    # split session into conditions
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

    # extract events from annotations

    event_id0 = {'BAD_data': 0, 'bad EPOCH': 100, 'BAD boundary': 100, 'EDGE boundary': 100}
    event_id1 = {'BAD_data': 1, 'bad EPOCH': 100, 'BAD boundary': 100, 'EDGE boundary': 100}

    # note: get_events() outputs
    # tmp[0],tmp[1], tmp[2]: events, epochs2keep, epochs2drop
    tmp = zeta.data.stim.get_events(raw_0, event_id0)
    events0 = tmp[0][tmp[1], :]
    tmp = zeta.data.stim.get_events(raw_1, event_id1)
    events1 = tmp[0][tmp[1], :]

    # events visualization
    if ShowFig:
        fig, ax = plt.subplot(211)
        color = {0: 'green', 100: 'red'}
        mne.viz.plot_events(events0, raw_0.info['sfreq'], raw_0.first_samp, color=color,
                            event_id=event_id0, axes=ax[0])
        mne.viz.plot_events(events1, raw_0.info['sfreq'], raw_0.first_samp, color=color,
                            event_id=event_id1, axes=ax[1])

    return raw_0, raw_1, events0, events1


def get_dataset_distress(data_folder, dataset_id, subject, ShowFig=False):
    """ Load Low Versus High Distress Tinnitus patient and control.

    Note here that the EEG data are in µV while MNE use V. Therefore scale
    is with a 1e6 factor and it could cause a problem for non-linear related MNE
    analysing. I advice to apply a 1e-6 factor in the future to make sure that
    everything is working fine with mne.
    For classification, it is adviced to keep data in µV.
    """

    # clean loop variable
    runs = []
    labels = []

    runs, labels, _, _ = load_sessions_raw(data_folder, dataset_id, subject)  # load all session from this subject

    raw_0 = mne.concatenate_raws(runs)
    raw_1 = []

    # rename table for event to annotations
    event_id0 = {'BAD_data': 0, 'bad EPOCH': 100, 'BAD boundary': 100, 'EDGE boundary': 100}

    # vizualization
    if ShowFig:
        scalings = dict(eeg=10e1)
        mne.viz.plot_raw(raw_0, scalings=scalings)

    # extract events from annotations
    if dataset_id is "Distress2010":
        event_id0 = {"Tinnitus": 1}
    elif dataset_id is "NormativeDB":
        event_id0 = {"Control": 0}

    event_id1 = None

    # generate artifical events for the future epoching
    start = 10    # (seconds) remove the beggining
    stop = None    # (seconds) remove the end
    interval = 2  # (seconds) between epochs
    events0 = mne.make_fixed_length_events(raw_0, id=1, duration=interval, start = start, stop = stop)
    events1 = []

    # events visualization
    if ShowFig:
        fig, ax = plt.subplot(211)
        color = {0: 'green', 100: 'red'}
        mne.viz.plot_events(events0, raw_0.info['sfreq'], raw_0.first_samp, color=color,
                            event_id=event_id0, axes=ax[0])
        mne.viz.plot_events(events1, raw_0.info['sfreq'], raw_0.first_samp, color=color,
                            event_id=event_id1, axes=ax[1])

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
    if dataset_id in ["raw_clean_32"]:
        # stack all subject's sessions
        for root, dirs, files in os.walk(os.path.join(data_folder, dataset_id)):
            for file in files:
                if file.startswith(subject):
                    filepath = os.path.join(root, file)
                    try:
                        runs.append(mne.io.read_raw_fif(filepath, verbose="ERROR"))  # stacks raw
                        # events=mne.read_events(filepath)  # might be usefull for some dataset

                        # WARNING: hardcoded label position could be troublesome in some dataset, check carefully
                        labels.append(file.split('_')[1])  # stacks session name
                        sessions_path.append(filepath)
                        subject_data_found = True
                    except:
                        print("Couldn't load subject " + subject + " session " + file.split('_')[1] + " at " + filepath)
                        bad_sessions_path.append(filepath)
    elif dataset_id in ["Distress2010","NormativeDB","Tinnitus_EEG"]:
        # stack all subject's sessions
        for root, dirs, files in os.walk(os.path.join(data_folder, dataset_id)):
            for file in files:
                if file.upper().startswith(subject):
                    filepath = os.path.join(root, file)
                    try:
                        data = _txt_to_numpy(filepath)
                        raw = _CreateRaw_T(data)
                        print(raw)
                        runs.append(raw)  # stacks raw

                        # WARNING: hardcoded label position could be troublesome in some dataset, check carefully
                        labels.append("rest")  # stacks session name

                        sessions_path.append(filepath)
                        subject_data_found = True
                    except:
                        print("Couldn't load subject " + subject + " at " + filepath)
                        bad_sessions_path.append(filepath)
    if len(runs) == 0:
        print("Couldn't load any session of subject " + subject + " in dataset " + dataset_id)
    return runs, labels, sessions_path, bad_sessions_path


def _sparse_info_from_file(filename, valid_id, separator="_"):
    """ Return a list of symptoms and conditions based on a list of valid_id keywords and return the filtered
    filename without the keywords.
    infos, filtered_filename=_sparse_info_from_file(filename, valid_id, separator="_")

        Parameters
    ----------
    filename : str
        filename without the extension ".*" (you can use filename.split(".")[0])
    valid_id : list of str
        the keywords corresponding to symptoms or conditions
    separator : str
        the separators between keywords (default: separator="_")

    Returns
    -------
    infos : list
        a list of symptoms sparsed from filename
    filtered_filename : str
        the filtered filename without the symptoms
    """
    infos = []

    all_keys = filename.upper().split(separator)
    for filekey in all_keys:
        if filekey in valid_id:
            infos.append(filekey)

    # rebuild file name from non valid_id keys
    filtered_filename = separator.join(
        [filekey for ind, filekey in enumerate(all_keys) if filekey not in valid_id]
    )

    return infos, filtered_filename


def _subjects_dict_to_pandas(dict_subjects):
    """Convert nested dictionary of subject/session to a MultiIndexed pandas structure

    To get subjects name list:
        df_subjects.index.get_level_values("subject").to_numpy()
    To get session file list:
        df_subjects.index.get_level_values("session").to_numpy()

    Example of Filtering with multiple criteria:
       ```
        # get Distress3 and Distress4 with NBN and "rest" paradigm only
        criterion1 = df_subjects["symptoms"].map(lambda x: {'distress': '3'} in x)
        criterion2 = df_subjects["symptoms"].map(lambda x: {'distress': '4'} in x)
        criterion3 = df_subjects["symptoms"].map(lambda x: "NBN" in x)
        criterion4 = df_subjects["paradigm"].map(lambda x: x == "rest")
        df_filtered = df_subjects[(criterion1 | criterion2) & criterion3 & criterion4]
        print(df_filtered)  # dataframe
        print(df_filtered).get_level_values("subject").to_list()  # subjects list only
        ```

    """
    df_subjects = pd.DataFrame.from_dict({(i, j): dict_subjects[i][j]
                                          for i in dict_subjects.keys()
                                          for j in dict_subjects[i].keys()},
                                         orient='index')

    df_subjects.index.names = ["subject", "session"]

    return df_subjects

def _txt_to_numpy(file):
    """Convert a txt file with space-separated numeric values to numpy structure.
    """

    # read file
    filo = open(file, "r")
    compt = 0
    signal = []

    # sparse each row
    for row in filo:
        listy = []
        listy = row.split(" ")  # space separated values
        listy = listy[1:]       # remove first space
        listy[-1] = listy[-1][:-2]
        # print listy
        loop = []
        for elm in listy:
            if not elm == "":
                loop.append(elm)
        loop = [float(loop[i]) for i in range(len(loop))]

        if not len(loop) == 19:
            print("File not consistent with dataset")
            print(file)
            print("_txt_to_numpy: n electrodes = %i" %len(loop))

        signal.append(loop)
        compt += 1

    signal = np.asanyarray(signal)
    signal = np.transpose(signal)

    # sub = 0
    # subj = subj[:9]
    # for sub in range(len(subj)):
    #     print(subj[sub])
    #     filo = open(subj[sub], "r")
    #
    #     compt = 0
    #     signal = []
    #     for row in filo:
    #         listy = []
    #         listy = row.split(" ")
    #
    #         listy[-1] = listy[-1][:-2]
    #         # print listy
    #         loop = []
    #         for elm in listy:
    #             if not elm == "":
    #                 loop.append(elm)
    #         loop = [float(loop[i]) for i in range(len(loop))]
    #         # print loop
    #         # print len(loop)
    #         if not len(loop) == 19:
    #             print(loop)
    #             print(len(loop))
    #         signal.append(loop)
    #         compt += 1
    #     print(compt)
    #     print(signal[0])
    #     print(signal[-1])
    #     signal = np.asanyarray(signal)
    #     print(len(signal[0]))
    #     signal = np.transpose(signal)
    #     print(len(signal[-1]))

    return signal


# Create Raw file
def _CreateRaw_T(data):
    """ Distress Tinnitus dataset (high and low distress)
    """
    ch_names = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8",
                "O1", "O2"]
    # ch_names=["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T3", "C3", "CZ", "C4", "T4", "T5", "P3", "PZ", "P4", "T6", "O1", "O2"]
    ch_types = ['eeg'] * len(ch_names)
    sfreq = 128
    montage = 'standard_1020'
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types, montage=montage, verbose=False)
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


def _CreateRaw_H(data):
    """ Distress Tinnitus dataset (control)
    """
    # ch_names = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
    ch_names = ["FP1", "FP2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6",
                "O1", "O2"]
    ch_types = ['eeg'] * len(ch_names)
    sfreq = 128
    montage = 'standard_1020'
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types, montage=montage, verbose=False)
    raw = mne.io.RawArray(data, info, verbose=False)
    return (raw)


if __name__ == '__main__':
    testDistress = False
    testNormativeDB = True
    # test Distress
    if testDistress:
        # example of loading dataset info and doing a panda query for specific subject within this dataset
        data_dir, output_dir = zeta.configuration.load_directories()
        dataset_name = "Distress2010"
        dict_subjects = get_subjects_info(data_dir, dataset_name)

        df_subjects = _subjects_dict_to_pandas(dict_subjects)

        print(df_subjects.index.get_level_values("subject").to_list())

        # example of query
        criterion1 = df_subjects["symptoms"].map(lambda x: {'distress': '3'} in x)
        criterion2 = df_subjects["symptoms"].map(lambda x: {'distress': '4'} in x)
        criterion3 = df_subjects["symptoms"].map(lambda x: "NBN" in x)
        criterion4 = df_subjects["paradigm"].map(lambda x: x == "rest")

        print(df_subjects[(criterion1 | criterion2) & criterion3 & criterion4])

        # get the raw data from the query

        subject=df_subjects[(criterion1 | criterion2) & criterion3 & criterion4].index.get_level_values("subject").to_list()[10]
        print(load_sessions_raw(data_dir, dataset_name, subject))
        raw_0, _ , events0, _ = get_raw(data_dir, dataset_name, subject)
        epochs = mne.Epochs(raw_0 ,events0, tmin=0, tmax=2, baseline=(None,0))

    # test Distress
    if testNormativeDB:
        # example of loading dataset info and doing a panda query for specific subject within this dataset
        data_dir, output_dir = zeta.configuration.load_directories()
        dataset_name = "NormativeDB"
        dict_subjects = get_subjects_info(data_dir, dataset_name)
        dict_subjects.keys()

        df_subjects = _subjects_dict_to_pandas(dict_subjects)

        print(df_subjects.index.get_level_values("subject").to_list())

        # example of query
        criterion1 = df_subjects["symptoms"].map(lambda x: {'distress': '0'} in x)
        criterion2 = df_subjects["gender"].map(lambda x: "M" in x)

        print(df_subjects[(criterion1 & criterion2)])

        # get the raw data from the query

        subject = df_subjects[(criterion1 & criterion2)].index.get_level_values("subject").to_list()[10]
        print(load_sessions_raw(data_dir, dataset_name, subject))
        raw_0, _ , events0, _ = get_raw(data_dir, dataset_name, subject)
        epochs = mne.Epochs(raw_0 ,events0, tmin=0, tmax=2, baseline=(None,0))
"""    assert False


    sub = 10
    print(subj[sub])
    filo = open(subj[sub], "r")
    compt = 0
    signal = []
    for row in filo:
        if compt == 0:
            print(row)
        listy = []
        listy = row.split(" ")
        listy = listy[1:]
        listy[-1] = listy[-1][:-2]
        # print listy
        loop = []
        for elm in listy:
            if not elm == "":
                loop.append(elm)
        loop = [float(loop[i]) for i in range(len(loop))]
        # print loop
        # print len(loop)
        signal.append(loop)
        compt += 1

    print(compt)

    sub = 0
    subj = subj[:9]
    for sub in range(len(subj)):
        print(subj[sub])
        filo = open(subj[sub], "r")

        compt = 0
        signal = []
        for row in filo:
            listy = []
            listy = row.split(" ")

            listy[-1] = listy[-1][:-2]
            # print listy
            loop = []
            for elm in listy:
                if not elm == "":
                    loop.append(elm)
            loop = [float(loop[i]) for i in range(len(loop))]
            # print loop
            # print len(loop)
            if not len(loop) == 19:
                print(loop)
                print(len(loop))
            signal.append(loop)
            compt += 1
        print(compt)
        print(signal[0])
        print(signal[-1])
        signal = np.asanyarray(signal)
        print(len(signal[0]))
        signal = np.transpose(signal)
        print(len(signal[-1]))
        RAW = _CreateRaw_T(signal)
        RAW.plot(scalings="auto")
"""
