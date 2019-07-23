"""Zeta analysis of ongoing EEG in frequency domain"""
import numpy as np
import mne
import matplotlib.pyplot as plt

def GFP(raw,events,
        tmin=-0.2,
        tmax=1,
        event_id=0,
        baseline=(None,0),
        iter_freqs=None,
        reject=None,
        verbose=None
        ):
    """ Global Field Power analysis from events. 
        Inspired from the tutorial of Denis A. Engemann:
        https://mne-tools.github.io/dev/auto_examples/time_frequency/plot_time_frequency_global_field_power.html
        Author: Louis Korczowski (louis.korczowski@gmail.com)
        Inherited License: BSD (3-clause)
    
    Parameters
    ----------
    raw : mne.io.raw object
        An instance of Raw of mne.
    events: array of int, shape (n_events, 3)
        The events typically returned by the read_events function. If some
        events don’t match the events of interest as specified by event_id,
        they will be marked as ‘IGNORED’ in the drop log.
    tmin,tmax : float (default=-0.2,1)
        Start and End time of selection in seconds.
    event_id : dict | int (default=0)
        The id of the event to consider. If dict, the keys can later be used to 
        access associated events. Example: dict(auditory=1, visual=3).
    baseline : tuple of length 2 (default=(None,0))
        The time interval to apply baseline correction. 
    iter_freqs : list of tuple | None (default=[('Theta', 4, 7),('Alpha', 8, 12),('Beta', 13, 25),('Gamma', 30, 45)])
        each frequency band used for the analysis
    reject : dict | None
        Rejection parameters based on peak-to-peak amplitude. Valid keys are 
        ‘grad’ | ‘mag’ | ‘eeg’ | ‘eog’ | ‘ecg’. If reject is None then no 
        rejection is done. 
        Example:
            ```
            reject = dict(grad=4000e-13, # T / m (gradiometers)
                  mag=4e-12, # T (magnetometers)
                  eeg=40e-6, # V (EEG channels)
                  eog=250e-6 # V (EOG channels)
                  )
            ```
    n_jobs : int (default: 1)
        Number of jobs to run in parallel.
    

    Returns
    ----------
    frequency_map : list of tuple
        each tuple as length 2 containing for each frequency band the following
        ((band, fmin, fmax), epochs.average())
    
    GFP : #TODO: implement output
        
    See Also
    --------
    
    References
    ----------
    [1]
    

    TODO: [] make condition epochs2keep as argument (length of the epoch)
    TODO: [] compatibility for different datasets
    
    """
    
    if iter_freqs==None:
        iter_freqs=[('Theta', 4, 7),('Alpha', 8, 12),('Beta', 13, 25),('Gamma', 30, 45)]
        
    frequency_map = list()
    for band, fmin, fmax in iter_freqs:
        # (re)load the data to save memory
        rawc=[]
        epochs=[]
        rawc = raw.copy()
        rawc.load_data(verbose=verbose)
        # bandpass filter and compute Hilbert
        rawc.filter(fmin, fmax, n_jobs=2,  # use more jobs to speed up.
                   l_trans_bandwidth=1,  # make sure filter params are the same
                   h_trans_bandwidth=1,  # in each band and skip "auto" option.
                   fir_design='firwin',
                   verbose=verbose)
        rawc.apply_hilbert(n_jobs=1, envelope=False,verbose=verbose)
    
        epochs = mne.Epochs(rawc, events, event_id, tmin, tmax, baseline=baseline,
                            reject=reject, preload=True,reject_by_annotation=0,
                            verbose=verbose)
        # remove evoked response and get analytic signal (envelope)
        epochs.subtract_evoked()  # for this we need to construct new epochs.
        epochs = mne.EpochsArray(
            data=np.abs(epochs.get_data()), info=epochs.info, tmin=epochs.tmin,
             verbose=verbose)
        # now average and move on
        frequency_map.append(((band, fmin, fmax), epochs.average()))
    
    fig, axes = plt.subplots(len(frequency_map), 1, figsize=(10, 7), sharex=True, sharey=True)
    colors = plt.get_cmap('winter_r')(np.linspace(0, 1, len(frequency_map)))
    for ((freq_name, fmin, fmax), average), color, ax in zip(
            frequency_map, colors, axes.ravel()[::-1]):
        times = average.times
        gfp = np.sum(average.data ** 2, axis=0)
        gfp = mne.baseline.rescale(gfp, times, baseline=baseline, verbose=verbose)
        ax.plot(times, gfp, label=freq_name, color=color, linewidth=2.5)
        ax.axhline(0, linestyle='--', color='grey', linewidth=2)
        ci_low, ci_up = mne.stats._bootstrap_ci(average.data, random_state=0,
                                      stat_fun=lambda x: np.sum(x ** 2, axis=0))
        ci_low = mne.baseline.rescale(ci_low, average.times, baseline=baseline, verbose=verbose)
        ci_up = mne.baseline.rescale(ci_up, average.times, baseline=baseline, verbose=verbose)
        ax.fill_between(times, gfp + ci_up, gfp - ci_low, color=color, alpha=0.3)
        ax.grid(True)
        ax.set_ylabel('GFP (K=%d)' % (average.nave))
        ax.annotate('%s (%d-%dHz)' % (freq_name, fmin, fmax),
                    xy=(0.95, 0.8),
                    horizontalalignment='right',
                    xycoords='axes fraction')
        ax.set_xlim(tmin, tmax)
    
    axes.ravel()[-1].set_xlabel('Time [s]')
    
    return frequency_map,fig

def TF(epochs,
       freqs=None,
       iter_freqs=None,
       ch_name=None,
       tmin=-0.2,
       tmax=1,
       baseline=(-0.2,0),
       mode='logratio',
       n_jobs=1,
       decim=1
       ):
    """Time-Frequency Analysis. 
    
    Author: Louis Korczowski (louis.korczowski@gmail.com)
    
    Parameters
    ----------
    epochs : mne.Epoch object
        An instance of selected Epochs of mne (only one condition)
    tmin,tmax : float (default=-0.2,1)
        Start and End time of selection in seconds.
    baseline : tuple of length 2 (default=(None,0))
        The time interval to apply baseline correction. 
    ch_name : str 
        A channel name for the plot example
    iter_freqs : list of tuple | None (default=[('Theta', 4, 7),('Alpha', 8, 12),('Beta', 13, 25),('Gamma', 30, 45)])
        each frequency band used for the analysis
    reject : dict | None
        Rejection parameters based on peak-to-peak amplitude. Valid keys are 
        ‘grad’ | ‘mag’ | ‘eeg’ | ‘eog’ | ‘ecg’. If reject is None then no 
        rejection is done. 
        Example:
            ```
            reject = dict(grad=4000e-13, # T / m (gradiometers)
                  mag=4e-12, # T (magnetometers)
                  eeg=40e-6, # V (EEG channels)
                  eog=250e-6 # V (EOG channels)
                  )
            ```
    n_jobs : int (default=1)
        Number of jobs to run in parallel.
    decim : int (default=1)
    
    

    Returns
    ----------
    frequency_map : list of tuple
        each tuple as length 2 containing for each frequency band the following
        ((band, fmin, fmax), epochs.average())
    
    GFP : #TODO: implement output
        
    See Also
    --------
    
    References
    ----------
    [1]
    

    TODO: [] make condition epochs2keep as argument (length of the epoch)
    TODO: [] compatibility for different datasets
    
    """
    if iter_freqs==None:
        iter_freqs=[('Theta', 4, 7),('Alpha', 8, 12),('Beta', 13, 25),('Gamma', 30, 45)]
        """
        [('Delta', 1, 4),('Theta', 4, 7),('Alpha-', 8, 11),('Alpha+', 11, 14),
        ('Beta-', 14, 19),('Beta+', 19, 25),('Gamma-', 25, 32),('Gamma+', 32, 40)]
        """
    if freqs is None:
        freqs = np.logspace(*np.log10([0.5, 45]), num=32)
        freqs =np.arange(0.5, 45, 0.5)
    if ch_name is None:
        ch_name='Fz'
    n_cycles = freqs / 2.  # different number of cycle per frequency
    powers, itc =  mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                            return_itc=True, decim=1, n_jobs=n_jobs)
    topomap_args = dict(sensors=False, baseline=baseline,
                        colorbar=False,tmin=tmin,tmax=tmax,mode=mode,show=False)
    
    # TF value on the topographic map
    fig1=powers.plot_topo(baseline=baseline, mode=mode, title='Average power')

    # TF value for an electrode example
    # TODO: REMOVE or EXPEND to all electrodes (not that usefull as fig1 do the job)
    fig2=powers.plot([powers.ch_names.index(ch_name)], baseline=baseline, mode=mode, title=ch_name)
    
    # topomap for each frequency band
    fig3, axis = plt.subplots(1, len(iter_freqs), figsize=(7, 4))
    it=0
    for band, fmin, fmax in iter_freqs:
        powers.plot_topomap(fmin=fmin, fmax=fmax, title=band, axes=axis[it],
                            **topomap_args)
        it=it+1
    mne.viz.tight_layout()

    return powers, itc,fig1,fig2,fig3
