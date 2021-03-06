"""Zeta Classif Vizualization"""

import sklearn.metrics as metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ShowClassificationResults(y,predicted, PipelineTitle='', ax=None):
    print(PipelineTitle+': AUC=%.2f, AP=%.2f, ACC=%.2f, f1=%.2f' % (metrics.roc_auc_score(y, predicted),
                                                                    metrics.precision_score(y, predicted),
                                                                    metrics.accuracy_score(y, predicted),
                                                                    metrics.f1_score(y, predicted)
                                                                    )
          )
    report = pd.DataFrame(metrics.classification_report(y,predicted, output_dict=True)).transpose()
    # print(report)
    if ax is not None:
        # TODO: plot auc ROC here
        print("zeta.viz.classif.ShowClassificationResults: Figure not configured")
    return report


def ShowRegressionResults(y,predicted,PipelineTitle='',ax=None):
    print(PipelineTitle+': R²=%.2f' % metrics.r2_score(y,predicted)+ ' r=%.2f' % np.sqrt(metrics.r2_score(y,predicted))+
          ' MAE=%.2f' % metrics.median_absolute_error(y, predicted)+ ' EV=%.2f' % metrics.explained_variance_score(y,predicted))

    if ax is not None:
        ax.clear()
        ax.scatter(y, predicted, edgecolors=(0, 0, 0))
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        ax.set_title(PipelineTitle)
        ax.text(0.25, 0.95,
                  r'$R^2$=%.2f, MAE=%.2f' % (
                  metrics.r2_score(y, predicted), metrics.median_absolute_error(y, predicted)),
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)


def plot_cv_indices(cv, X, y, group=[], ax=None, n_splits=5, lw=10,
                    cmap_data = plt.cm.Paired,
                    cmap_cv = plt.cm.coolwarm):
    """Create a sample plot for indices of a cross-validation object."""
    if group == []:
        group=[x for x in range(0,len(y))]  # by default give the indice of the epoch

    if ax==None:
        fig, ax = plt.subplots()

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    # Plot the data classes and groups at the end
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=y, marker='_', lw=lw, cmap=cmap_data)

    ax.scatter(range(len(X)), [ii + 2.5] * len(X),
               c=group, marker='_', lw=lw, cmap=plt.cm.hsv)

    # Formatting
    yticklabels = list(range(n_splits)) + ['class', 'group']
    ax.set(yticks=np.arange(n_splits+2) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits+2.2, -.2], xlim=[0, len(y)])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax
