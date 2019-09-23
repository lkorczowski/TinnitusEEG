"""Zeta Classification pipeline estimations"""

from sklearn.base import BaseEstimator, TransformerMixin
import sklearn
import pyriemann
import numpy as np
import mne
from pyriemann.estimation import XdawnCovariances
from pyriemann.spatialfilters import Xdawn
import pandas as pd


def CreatesFeatsPipeline(pipe_name, init_params=None):
    """ load pre-existing pipelines
    """
    pipeline = []
    if pipe_name == 'cla_ERP_TS_LR':
        # pipeline using Xdawn with MDM
        pipeline = sklearn.pipeline.Pipeline([
            ('xdawn', pyriemann.estimation.XdawnCovariances())
            , ('TS', pyriemann.tangentspace.TangentSpace())
            , ('lr', sklearn.linear_model.LogisticRegression())
        ])
    elif pipe_name == 'cla_ERP_LR':
        pipeline = sklearn.pipeline.Pipeline([
            ('preproc', Epochs2signals())
            , ('xdawn', pyriemann.estimation.XdawnCovariances())
            , ('TS', pyriemann.tangentspace.TangentSpace())
            , ('lr', sklearn.linear_model.LogisticRegression())
        ])
    elif pipe_name == 'reg_ERP':
        # pipeline using Xdawn in the tangent space (regression)
        pipeline = sklearn.pipeline.Pipeline([
            ('preproc', Epochs2signals())
            , ('xdawn', XdawnCovariancesRegression())
            , ('TS', pyriemann.tangentspace.TangentSpace())
            , ('LASSO', sklearn.linear_model.LassoCV())
        ])
    elif pipe_name == 'reg_ERP_svr':
        # pipeline using Xdawn in the tangent space
        pipeline = sklearn.pipeline.Pipeline([
            ('preproc', Epochs2signals())
            , ('xdawn', XdawnCovariancesRegression())
            , ('TS', pyriemann.tangentspace.TangentSpace())
            , ('LASSO',
               sklearn.model_selection.GridSearchCV(sklearn.svm.SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1), cv=5,
                                                    param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                                                                "gamma": np.logspace(-2, 2, 5)}))
        ])

    elif pipe_name == 'reg_FilterBank':
        f_list = range(len(init_params['preproc__filters']))
        pipFreqs = []
        for freq in f_list:
            pipFreqs.append(("freq" + str(freq), sklearn.pipeline.Pipeline([
                ('CospSelector', CospSelector(f_list=[freq]))
                , ('Cov', pyriemann.estimation.Covariances(estimator='lwf'))
                #                 ,('xdawn',XdawnCovariancesRegression(nfilter=8,estimator='lwf',xdawn_estimator='lwf',bins=[0,32,72,100]))
                , ('SPOC', pyriemann.spatialfilters.SPoC(nfilter=20, log=False))
                #                 ,('TS',pyriemann.tangentspace.TangentSpace())
                , ('cosp2Feats', Cosp2feats())
            ])
                             )
                            )
        union = sklearn.pipeline.FeatureUnion(pipFreqs)

        pipeline = sklearn.pipeline.Pipeline([
            ('preproc', Epochs2signals())
            , ('union', union)
            , ('LASSO', sklearn.linear_model.LassoCV())
        ])

    elif pipe_name == 'reg_SPOC':
        pipeline = sklearn.pipeline.Pipeline([
            ('preproc', Epochs2signals())
            , ('Cov', pyriemann.estimation.Covariances())
            , ('SPOC', pyriemann.spatialfilters.SPoC())
            , ('TS', pyriemann.tangentspace.TangentSpace())
            , ('LASSO', sklearn.linear_model.LassoCV())
        ])
    else:
        print('no pipeline recognized')
        assert False

    # initialize parameters of the pipeline
    if init_params is not None:
        pipeline.set_params(**init_params)  # initialize the parameters
    else:
        print('CreatesFeatsPipeline: ' + pipe_name + ' not initialized!')

    return pipeline


class Epochs2signals(BaseEstimator, TransformerMixin):
    """Filters and crops epochs.
    
    Parameters
    ----------

    filters: list of list (defaults [[1, 35]])
        bank of bandpass filter to apply.

    events: List of str | None (default None)
        event to use for epoching. If None, default to all events defined in
        the dataset.

    tmin: float (default 0.0)
        Start time (in second) of the epoch, relative to the dataset specific
        task interval e.g. tmin = 1 would mean the epoch will start 1 second
        after the begining of the task as defined by the dataset.

    tmax: float | None, (default None)
        End time (in second) of the epoch, relative to the begining of the
        dataset specific task interval. tmax = 5 would mean the epoch will end
        5 second after the begining of the task as defined in the dataset. If
        None, use the dataset value.

    channels: list of str | None (default None)
        list of channel to select. If None, use all EEG channels available in
        the dataset.

    resample: float | None (default None)
        If not None, resample the eeg data with the sampling rate provided.

    See Also
    --------
    """

    def __init__(self, filters=([1, 35],), events=None, tmin=-0.2, tmax=1.0,
                 channels=None, resample=None, epochsinfo=None, epochstmin=None,
                 baseline=None):
        self.filters = filters
        self.resample = resample

        if (tmax is not None):
            if tmin >= tmax:
                raise (ValueError("tmax must be greater than tmin"))

        self.tmin = tmin
        self.tmax = tmax
        self.epochsinfo = epochsinfo
        self.epochstmin = epochstmin
        self.baseline = baseline

    def fit(self, X, y=None):
        """Fit.

        Do nothing. For compatibility purpose.
        

        Returns
        -------
        self : Epochs2signals instance
            The Epochs2signals instance.
        """
        return self

    def transform(self, X):
        """Convert epochs from mne format to numpy array
        """
        if str(type(X)) == "<class 'mne.epochs.Epochs'>":
            epochs = X.copy()
        elif str(type(X)) == "<class 'numpy.ndarray'>":
            # build mne structure (need epochs as input)
            epochs = mne.EpochsArray(X, info=self.epochsinfo, tmin=self.epochstmin)
        X = []
        for bandpass in self.filters:
            fmin, fmax = bandpass
            # filter epoched data

            epochs.filter(l_freq=fmin, h_freq=fmax, verbose=True)
            if self.resample is not None:
                epochs.resample(sfreq=self.resample)

            # crop epochs (important for removing edge effects)
            epochs.crop(tmin=self.tmin, tmax=self.tmax)

            if self.baseline is not None:
                epochs.apply_baseline(self.baseline, verbose=False)

            # MNE is in V, rescale to have uV
            X.append(1e6 * epochs.get_data())

        # if only one band, return a 3D array, otherwise return a 4D
        if len(self.filters) == 1:
            X = X[0]
        else:
            X = np.array(X).transpose((1, 2, 3, 0))

        return X


class CospSelector(BaseEstimator, TransformerMixin):
    """select meaningfull covariances from 4D matrices
    
    This method will return a 4-d array (or 3-d array) from a symmetric 4-d array (e.g
    a cospectrum covariance matrice estimation). It is very usefull when facing
    with multidimentional input that needs to be converted.
    By default this estimator does nothing. If f_list is only one number, it will
    convert to 3-d matrices instead.

    Parameters
    ----------
    f_list : list | None , (default None)
        the selected frequency indices (default ALL)

    See Also
    --------
    Covariances
    HankelCovariances
    Cosp2feats
    """

    def __init__(self, f_list=None):
        """Init."""
        self.f_list = f_list

    def fit(self, X, y=None):
        """Fit.

        Do nothing. For compatibility purpose.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels,n_channels, n_samples)
            ndarray of trials.
        y : ndarray shape (n_trials,)
            labels corresponding to each trial, not used.

        Returns
        -------
        self : Cosp2feats instance
            The Cosp2feats instance.
        """
        return self

    def transform(self, X):
        """Estimate the cospectral covariance matrices.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels, n_freq)
            ndarray of trials with second and third dimension symetricals

        Returns
        -------
        out : ndarray, shape (n_trials, n_channels*(n_channels+1)*n_freq/2)
            ndarray of features from covariances
        """
        _, _, _, Nf = X.shape
        if self.f_list == None:
            self.f_list = range(Nf)
        return np.squeeze(X[:, :, :, self.f_list])


class Cosp2feats(BaseEstimator, TransformerMixin):
    """transform 4D matrices to features

    This method will return a 2-d array from a symmetric 4-d array (e.g
    a cospectrum covariance matrice estimation). It is very usefull when facing
    with multidimentional input that needs to be converted.

    Parameters
    ----------
    f_list : list | None , (default None)
        the selected frequency indices (default ALL)

    See Also
    --------
    Covariances
    HankelCovariances
    Coherences
    """

    def __init__(self, f_list=None):
        """Init."""
        self.f_list = f_list

    def fit(self, X, y=None):
        """Fit.

        Do nothing. For compatibility purpose.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels,n_channels, n_samples)
            ndarray of trials.
        y : ndarray shape (n_trials,)
            labels corresponding to each trial, not used.

        Returns
        -------
        self : Cosp2feats instance
            The Cosp2feats instance.
        """
        return self

    def transform(self, X):
        """Estimate the cospectral covariance matrices.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels, n_freq)
            ndarray of trials with second and third dimension symetricals

        Returns
        -------
        out : ndarray, shape (n_trials, n_channels*(n_channels+1)*n_freq/2)
            ndarray of features from covariances
        """
        if X.ndim == 4:
            Nt, Ne, _, Nf = X.shape
        else:
            Nt, Ne, _ = X.shape
            Nf = 1

        out = []
        iu1 = np.triu_indices(Ne)

        if self.f_list == None:
            self.f_list = range(Nf)

        for i in range(Nt):
            tmp = []
            if Nf > 1:
                for f in self.f_list:
                    S = X[i, :, :, f]
                    feats = S[iu1]
                    tmp.append(feats)
            else:
                S = X[i, :, :]
                feats = S[iu1]
                tmp.append(feats)
            out.append(np.asarray(np.array(tmp).reshape(-1)))

        return np.array(out)


class XdawnCovariancesRegression(XdawnCovariances):
    """Estimate special form covariance matrix for ERP combined with Xdawn.

    Estimation of special form covariance matrix dedicated to ERP processing
    combined with Xdawn spatial filtering. This is similar to `ERPCovariances`
    but data are spatially filtered with `Xdawn`. A complete descrition of the
    method is available in [1].

    The advantage of this estimation is to reduce dimensionality of the
    covariance matrices efficiently.

    Parameters
    ----------
    nfilter: int (default 4)
        number of Xdawn filter per classes.
    applyfilters: bool (default True)
        if set to true, spatial filter are applied to the prototypes and the
        signals. When set to False, filters are applied only to the ERP prototypes
        allowing for a better generalization across subject and session at the
        expense of dimensionality increase. In that case, the estimation is
        similar to ERPCovariances with `svd=nfilter` but with more compact
        prototype reduction.
    classes : list of int | None (default None)
        list of classes to take into account for prototype estimation.
        If None (default), all classes will be accounted.
    estimator : string (default: 'scm')
        covariance matrix estimator. For regularization consider 'lwf' or 'oas'
        For a complete list of estimator, see `utils.covariance`.
    xdawn_estimator : string (default: 'scm')
        covariance matrix estimator for xdawn spatial filtering.
    baseline_cov : array, shape(n_chan, n_chan) | None (default)
        baseline_covariance for xdawn. see `Xdawn`.
    bins : list, len(n_bins+1) | None (default)
        for bagging the labels y into bins. Usefull when the labels are continuous
        and want to use xdawn with multi-class.

    See Also
    --------
    ERPCovariances
    Xdawn

    References
    ----------
    [1] Barachant, A. "MEG decoding using Riemannian Geometry and Unsupervised
        classification."
    """

    def __init__(self, nfilter=4, applyfilters=True, classes=None,
                 estimator='scm', xdawn_estimator='scm', baseline_cov=None,
                 bins=None):
        """Init."""
        self.applyfilters = applyfilters
        self.estimator = estimator
        self.xdawn_estimator = xdawn_estimator
        self.classes = classes
        self.nfilter = nfilter
        self.baseline_cov = baseline_cov
        self.bins = bins

    def fit(self, X, y):
        """Fit.

        Estimate spatial filters and prototyped response for each classes.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.
        y : ndarray shape (n_trials,)
            labels corresponding to each trial.

        Returns
        -------
        self : XdawnCovariances instance
            The XdawnCovariances instance.
        """

        yb = continuous2discrete(y, self.bins)
        self.Xd_ = Xdawn(nfilter=self.nfilter, classes=self.classes,
                         estimator=self.xdawn_estimator,
                         baseline_cov=self.baseline_cov)

        self.Xd_.fit(X, yb)
        self.P_ = self.Xd_.evokeds_

        return self


###############################################################################

def continuous2discrete(y, bins):
    """convert continuous labels to discrete
    
        Usefull when facing with method only using discrete number of class.
    """
    # TODO: not elegant at all but efficient and tested. Can be implemented.
    if bins is None:
        yb = y
    else:
        if len(bins) == 3:  # 2 classes
            yb = [0 if num <= bins[1] else 1 for num in y]
        elif len(bins) == 4:  # 3 classes
            yb = [0 if num <= bins[1] else 1 if bins[1] <= num < bins[2] else 2 for num in y]  # 3 class
        elif len(bins) == 5:  # 4 classes
            yb = [0 if num <= bins[1] else 1 if bins[1] <= num < bins[2] else 2 if bins[2] <= num < bins[3]
            else 3 for num in y]  # 3 class
        elif len(bins) == 6:  # 5 classes
            yb = [0 if num <= bins[1] else 1 if bins[1] <= num < bins[2] else 2 if bins[2] <= num < bins[3]
            else 3 if bins[3] <= num < bins[4] else 4 for num in y]  # 3 class
        else:
            raise ValueError("wrong number of bins")
    assert ("class in bins range with not a single epoch", len(bins) - 1 == len(np.unique(yb)))
    return yb


class ERPCovariancesRegression(pyriemann.estimation.ERPCovariances):
    """Estimate special form covariance matrix for ERP. (ALTERNATIVE for regression)

    Estimation of special form covariance matrix dedicated to ERP processing.
    For each class in range, a prototyped response is obtained by average across trial :

    .. math::
        \mathbf{P} = \\frac{1}{N} \sum_i^N \mathbf{X}_i

    and a super trial is build using the concatenation of P and the trial X :

    .. math::
        \mathbf{\\tilde{X}}_i =  \left[
                                 \\begin{array}{c}
                                 \mathbf{P} \\\\
                                 \mathbf{X}_i
                                 \end{array}
                                 \\right]

    This super trial :math:`\mathbf{\\tilde{X}}_i` will be used for covariance
    estimation.
    This allows to take into account the spatial structure of the signal, as
    described in [1].

    Parameters
    ----------
    classes : list of range | None (default None)
        list of range of class to take into account for prototype estimation.
        If None (default), all classes will be accounted for an unique ERP.
    estimator : string (default: 'scm')
        covariance matrix estimator. For regularization consider 'lwf' or 'oas'
        For a complete list of estimator, see `utils.covariance`.
    svd : int | None (default None)
        if not none, the prototype responses will be reduce using a svd using
        the number of components passed in svd.

    See Also
    --------
    Covariances
    XdawnCovariances
    CospCovariances
    HankelCovariances

    References
    ----------
    [1] A. Barachant, M. Congedo ,"A Plug&Play P300 BCI Using Information
    Geometry", arXiv:1409.0107, 2014.

    [2] M. Congedo, A. Barachant, A. Andreev ,"A New generation of
    Brain-Computer Interface Based on Riemannian Geometry", arXiv: 1310.8115.
    2013.

    [3] A. Barachant, M. Congedo, G. Van Veen, C. Jutten, "Classification de
    potentiels evoques P300 par geometrie riemannienne pour les interfaces
    cerveau-machine EEG", 24eme colloque GRETSI, 2013.
    """

    def fit(self, X, y):
        """Fit.

        Estimate the Prototyped response for each classes.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.
        y : ndarray shape (n_trials,)
            labels corresponding to each trial.

        Returns
        -------
        self : ERPCovariances instance
            The ERPCovariances instance.
        """
        if self.classes is not None:
            classes = self.classes
        else:
            classes = [pd.Interval(left=y.min(), right=y.max(), closed='both')]

        self.P_ = []
        for c in classes:
            # Prototyped responce for each class
            P = np.mean(X[[i in c for i in y], :, :], axis=0)

            # Apply svd if requested
            if self.svd is not None:
                U, s, V = np.linalg.svd(P)
                P = np.dot(U[:, 0:self.svd].T, P)

            self.P_.append(P)

        self.P_ = np.concatenate(self.P_, axis=0)
        return self
