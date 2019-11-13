"""Analysis for classification results and specific pipelines with figures output

The aim of the following function are to detect specific pattern used by the classifiers, mainly in spatial and spectral contributions
"""

import mne
import zeta
from scipy.signal import welch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import os

def spatio_spectral_patterns(epochs,y, n_components = 4, output_dir="", test_name = "test"):
    """Computes the Common Spatial Pattern (CSP) on all data and print the most discriminant feature and
    performs a simple CSP+Logistic Regression Classification. The results will be stacked for all subjects
    at the end of pipeline_1.
    code inspired by Alexandre Barachant's Kaggle
    https://www.kaggle.com/alexandrebarachant/common-spatial-pattern-with-mne
    another reading for CSP decoding
    https://www.nmr.mgh.harvard.edu/mne/dev/auto_examples/decoding/plot_decoding_csp_eeg.html
    """
    auc = []
    X = epochs.get_data()
    # run CSP
    zeta.util.blockPrint()  # to clean the terminal
    csp = mne.decoding.CSP(reg='ledoit_wolf')
    csp.fit(X, y)
    zeta.util.enablePrint()  # restore print

    # compute spatial filtered spectrum for each components
    for indc in range(n_components):
        po = []
        for x in X:
            f, p = welch(np.dot(csp.filters_[indc, :].T, x), int(epochs.info['sfreq']), nperseg=256)
            po.append(p)
        po = np.array(po)

        # prepare topoplot
        _, epos, _, _, _ = mne.viz.topomap._prepare_topo_plot(epochs, 'eeg', None)

        # plot first pattern
        pattern = csp.patterns_[indc, :]
        pattern -= pattern.mean()
        ix = np.argmax(abs(pattern))

        # the parttern is sign invariant.
        # invert it for display purpose
        if pattern[ix] > 0:
            sign = 1.0
        else:
            sign = -1.0

        fig, ax_topo = plt.subplots(1, 1, figsize=(12, 4))
        title = 'Spatial Pattern'
        fig.suptitle(title, fontsize=14)
        img, _ = mne.viz.topomap.plot_topomap(sign * pattern, epos, axes=ax_topo, show=False)
        divider = make_axes_locatable(ax_topo)
        # add axes for colorbar
        ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(img, cax=ax_colorbar)

        # plot spectrum
        fix = (f > 1) & (f < 35)
        ax_spectrum = divider.append_axes('right', size='300%', pad=1.2)
        ax_spectrum.plot(f[fix], np.log(po[y == 0][:, fix].mean(axis=0).T), '-r', lw=2)
        ax_spectrum.plot(f[fix], np.log(po[y == 1][:, fix].mean(axis=0).T), '-b', lw=2)

        ax_spectrum.plot(f[fix], np.log(np.median(po[y == 0][:, fix], axis=0).T), '-r', lw=0.5)
        ax_spectrum.plot(f[fix], np.log(np.min(po[y == 0][:, fix], axis=0).T), '--r', lw=0.5)
        ax_spectrum.plot(f[fix], np.log(np.max(po[y == 0][:, fix], axis=0).T), '--r', lw=0.5)

        ax_spectrum.plot(f[fix], np.log(np.median(po[y == 1][:, fix], axis=0).T), '-b', lw=0.5)
        ax_spectrum.plot(f[fix], np.log(np.min(po[y == 1][:, fix], axis=0).T), '--b', lw=0.5)
        ax_spectrum.plot(f[fix], np.log(np.max(po[y == 1][:, fix], axis=0).T), '--b', lw=0.5)

        ax_spectrum.set_xlabel('Frequency (Hz)')
        ax_spectrum.set_ylabel('Power (dB)')
        plt.grid()
        plt.legend(['A', 'B'])

        # plt.show()
        plt.savefig( os.path.join(output_dir, 'spatial_pattern_subject_' + test_name +
                    '_c' + str(indc) + '.png'), bbox_inches='tight')

    # run cross validation
    #zeta.util.blockPrint()  # to have a clean terminal
    """
    clf = sklearn.pipeline.make_pipeline(mne.decoding.CSP(n_components=n_components),
                                         sklearn.linear_model.LogisticRegression(solver="lbfgs"))
    cv = sklearn.model_selection.StratifiedKFold(n_splits=5)
    auc.append(sklearn.model_selection.cross_val_score(clf, X, y, cv=cv, scoring='roc_auc').mean())
    zeta.util.enablePrint()
    

    print(test_name + " : AUC cross val score : %.3f" % (auc[-1]))
    """