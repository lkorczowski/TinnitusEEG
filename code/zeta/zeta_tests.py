""" Trash script for non-official unit test of the modules.
You can use it has you want but keep it as simple as possible.

"""
import zeta

lol=None

for i in range(5):
    try:
        if i==2:
            assert False
        elif i==3:
            lol=zeta.data.datasets.load_sessions_raw("/Volumes/Ext/data/Zeta/", "raw_clean_32", "x")[0]
            print(lol)
        else:
            print(i)
    except AssertionError:
        print("unit test went wrong %i"%(i))
    except:
        print("something went wrong here %i"%(i))

"""save in case
# ----------------------------
# RAW DATA LOADING AND PREPROCESSING
# %%--------------------------
fig_dir_sub = fig_dir + subject + os.path.sep
if not os.path.exists(fig_dir_sub):
    os.makedirs(fig_dir_sub)  # create results directory if needed

# load subject data


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
"""