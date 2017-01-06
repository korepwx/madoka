# -*- coding: utf-8 -*-
import itertools
import re
import unicodedata
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import r2_score
from sklearn.utils.multiclass import unique_labels

from madoka.utils import adaptive_density

__all__ = [
    'safe_filename',
    'replace_zero_divisor', 'classfication_report_data_frame',
    'normalize_regression_report_args', 'regression_report_data_frame',
    'plot_pdf_cdf', 'plot_regression_outline'
]

# value to replace the divisor when zero is evolved.
DIV_ZERO_EPS = 1e-5


def safe_filename(name):
    """Get a safe filename according to specified name."""
    name = (unicodedata.normalize('NFKD', name).
            encode('ascii', 'ignore').
            decode('utf-8'))
    name = re.sub('[^\w\s-]', '', name).strip().lower()
    name = re.sub('[-\s]+', '-', name)
    return name


def replace_zero_divisor(values):
    """Replace zero values for using as divisor."""
    mask = (np.abs(values) > DIV_ZERO_EPS).astype(np.int32)
    return (values * mask) + DIV_ZERO_EPS * (1 - mask)


def classfication_report_data_frame(y_true, y_pred, labels=None,
                                    target_names=None):
    """Generate the classification report as pandas data frame.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth (correct) target values.

    y_pred : numpy.ndarray
        Estimated targets as returned by a classifier.

    labels : numpy.ndarray
        If specified, will compute the precision-recall of only these labels.

    target_names : list
        If specified, will use this as the class names, instead of "Class ??".

    Returns
    -------
    pd.DataFrame
    """
    if labels is None:
        labels = unique_labels(y_true, y_pred)
    if target_names is None:
        target_names = [str(i) for i in labels]

    p, r, f1, s = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels
    )

    # compute the average of these scores.
    p_avg = np.average(p, weights=s)
    r_avg = np.average(r, weights=s)
    f1_avg = np.average(f1, weights=s)
    s_sum = np.sum(s)

    # compose the data frame
    def concat(a, b):
        return np.concatenate([a, [b]])
    data = OrderedDict([
        ('Precision', concat(p, p_avg)),
        ('Recall', concat(r, r_avg)),
        ('F1-Score', concat(f1, f1_avg)),
        ('Support', concat(s, s_sum)),
    ])

    return pd.DataFrame(data=data, columns=list(data.keys()),
                        index=target_names + ['avg / total'])


def normalize_regression_report_args(truth, predict, label=None, targets=None,
                                     per_target=True):
    """Normalize the arguments for a regression report.

    The dimension of truth and predict will be normalized to 2d.

    Parameters
    ----------
    truth : numpy.ndarray
        Ground truth (correct) target values.

    predict : numpy.ndarray
        Estimated targets as returned by a regressor.

    label : numpy.ndarray
        Label for each testing data, as an auxiliary information to
        demonstrate the regression result. (Optional)

    targets : numpy.ndarray
        Names of the targets, as an auxiliary information to demonstrate
        the regression result. (Optional)

        The shape of this array must match the shape of each regression target,
        i.e., `targets.shape == truth.shape[1:]`.

    per_target : bool
        If True, will compute per-target statistics.
        Otherwise will only compute the total statistics.

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        The tuple of normalized (truth, predict, label, targets).
    """
    # check the arguments
    if predict.shape != truth.shape:
        raise TypeError('Shape of `predict` does not match `truth`.')
    if label is not None and len(label) != len(truth):
        raise TypeError('Size of `label` != size of `truth`.')

    # generate the target labels.
    if not per_target:
        targets = None
    else:
        if len(truth.shape) == 1:
            if targets is None:
                targets = ['0']
            else:
                if not isinstance(targets, np.ndarray):
                    targets = np.asarray(targets)
                if len(targets) != 1 or not isinstance(targets[0], str):
                    raise TypeError('Shape of `targets` does not match '
                                    '`truth`.')
        else:
            if targets is None:
                targets = []
                indices_it = (range(i) for i in truth.shape[1:])
                for indices in itertools.product(*indices_it):
                    targets.append('-'.join(str(s) for s in indices))
                targets = np.asarray(targets)
            else:
                if not isinstance(targets, np.ndarray):
                    targets = np.asarray(targets)
                if targets.shape != truth.shape[1:]:
                    raise TypeError('Shape of `targets` does not match '
                                    '`truth`.')
                targets = targets.reshape([-1])

    # flatten the dimensions in truth and target data to match the targets
    truth = truth.reshape([len(truth), -1])
    predict = predict.reshape([len(predict), -1])

    return truth, predict, label, targets


def regression_report_data_frame(truth, predict, label=None, targets=None,
                                 per_target_summary=True):
    """Generate the regression report as pandas data frame.

    Parameters
    ----------
    truth : numpy.ndarray
        Ground truth (correct) target values.

    predict : numpy.ndarray
        Estimated targets as returned by a regressor.

    label : numpy.ndarray
        Label for each testing data, as an auxiliary information to
        demonstrate the regression result. (Optional)

    targets : numpy.ndarray
        Names of the targets, as an auxiliary information to demonstrate
        the regression result. (Optional)

        The shape of this array must match the shape of each regression target,
        i.e., `targets.shape == truth.shape[1:]`.

    per_target_summary : bool
        If True, will generate per-target summary.
        Otherwise will only generate the total summary.

    Returns
    -------
    pd.DataFrame
    """
    def mkrow(y_true, y_pred):
        return np.asarray([
            mean_squared_error(y_true, y_pred),
            mean_absolute_error(y_true, y_pred),
            r2_score(y_true, y_pred, multioutput='uniform_average'),
            explained_variance_score(y_true, y_pred),
        ])

    columns = (
        'Mean Squared Error', 'Mean Absolute Error',
        'R2 Score', 'Explained Variance Score',
    )

    truth, predict, label, targets = normalize_regression_report_args(
        truth, predict, label, targets, per_target_summary
    )

    # compose the data and the index
    data = []
    if targets is None:
        if label is None:
            data = [mkrow(truth, predict)]
            index = pd.Index(['total'])
        else:
            ulabels = list(unique_labels(label))
            for lbl in ulabels:
                mask = (label == lbl)
                data.append(mkrow(truth[mask], predict[mask]))
            data.append(mkrow(truth, predict))
            ulabels.append('total')
            index = pd.Index(ulabels, name='Label')
    else:
        if label is None:
            data = [mkrow(truth[:, i], predict[:, i])
                    for i in range(len(targets))]
            data.append(mkrow(truth, predict))
            index = pd.Index(list(targets) + ['total'], name='Target')
        else:
            target_index = []
            label_index = []
            for i in range(len(targets)):
                for lbl in list(unique_labels(label)):
                    mask = (label == lbl)
                    target_index.append(targets[i])
                    label_index.append(lbl)
                    data.append(mkrow(truth[mask][:, i], predict[mask][:, i]))
            target_index.append('')
            label_index.append('total')
            data.append(mkrow(truth, predict))
            index = pd.MultiIndex.from_tuples(
                [tuple(v) for v in zip(target_index, label_index)],
                names=['Target', 'Label']
            )

    return pd.DataFrame(data=data, columns=columns, index=index)


def plot_pdf_cdf(values, ax, max_bins=500, pdf_color='navy',
                 cdf_color='orangered', y_tick_num=10):
    """Plot the PDF as well as CDF in one figure.

    Parameters
    ----------
    values : numpy.ndarray
        The values of which PDF and CDF should be plotted.

    ax
        Matplotlib axes object.

    max_bins : int
        Maximum number of bins in histogram.
        The actual bins may even smaller than this if the data is too few.

    pdf_color : str
        Color of the PDF curve and the PDF y-axis.

    cdf_color : str
        Color of the CDF curve and the CDF y-axis.

    y_tick_num : int
        Number of ticks on the y-axis.
    """
    bins = min(max_bins, int(len(values) * .1))
    pdf, edges, left, right = adaptive_density(values, bins=bins)
    edges_mid = (edges[1:] + edges[:-1]) * .5
    cdf = (np.cumsum(pdf) + left) / (np.sum(pdf) + left + right)

    # determine the limits and ticks of each axis
    v_min, v_max = edges_mid[0], edges_mid[-1]
    p_min, p_max = np.min(pdf), np.max(pdf)
    c_min, c_max = 0.0, 1.0

    # plot the PDF, with y axes at the left
    ax.plot(edges_mid, pdf, color=pdf_color, linewidth=2.0)
    ax.set_ylabel('PDF', color=pdf_color)
    for tl in ax.get_yticklabels():
        tl.set_color(pdf_color)

    # plot the CDF, with y axes at the right
    ax2 = ax.twinx()
    ax2.plot(edges_mid, cdf, color=cdf_color, linewidth=2.0)
    ax2.set_ylabel('CDF', color=cdf_color)
    for tl in ax2.get_yticklabels():
        tl.set_color(cdf_color)

    # tweak the viewport of axes
    ax.set(adjustable='box-forced',
           xlim=[v_min, v_max],
           ylim=[p_min, p_max],
           yticks=np.linspace(p_min, p_max, y_tick_num + 1))
    ax2.set(adjustable='box-forced',
            xlim=[v_min, v_max],
            ylim=[c_min, c_max],
            yticks=np.linspace(c_min, c_max, y_tick_num + 1))
    ax.grid()


def plot_regression_outline(ax, truth, predict, bins=100, data_color='navy',
                            err_color='orangered', r2_color='forestgreen',
                            y_tick_num=10):
    """Plot the outline of regression result."""
    assert(len(truth.shape) == 1)
    assert(truth.shape == predict.shape)
    eps = np.finfo(truth.dtype).eps

    # compute the histogram of truth data
    hist, edges, _, _ = adaptive_density(truth, bins=bins)
    edges_mid = (edges[:-1] + edges[1:]) * 0.5
    edge_w = edges[1] - edges[0]
    edges_2d = edges.reshape([1, -1])
    truth_2d = truth.reshape([-1, 1])

    # compute which bin does every point falls in
    truth_index = np.argmax(
        ((truth_2d > edges_2d[:, :-1] - eps) &
         (truth_2d < edges_2d[:, 1:] + eps)).astype(np.int32),
        axis=1
    )

    # now compute the relative absolute errors of each bin
    errors = []
    for i in range(len(edges) - 1):
        mask = truth_index == i
        if np.any(mask):
            masked_truth = replace_zero_divisor(truth[mask])
            err = (predict[mask] - masked_truth) / masked_truth
            errors.append(np.average(np.abs(err)))
        else:
            errors.append(0.0)
    errors = np.asarray(errors)

    # now compute the r2 score of each bin
    r2_scores = []
    for i in range(len(edges) - 1):
        mask = truth_index == i
        if np.any(mask):
            r2_scores.append(r2_score(truth[mask], predict[mask]))
        else:
            r2_scores.append(1.0)
    r2_scores = np.asarray(r2_scores)

    # determine the limits and ticks of each axis
    x_min, x_max = edges[0], edges[-1]
    d_min, d_max = np.min(hist), np.max(hist)
    r_min, r_max = np.min(r2_scores), np.max(r2_scores)
    e_min, e_max = np.min(errors), np.max(errors)

    # make the axes
    ax2 = ax.twinx()
    ax0 = ax.twinx()

    # plot the data histogram
    ax0.bar(edges[:-1], hist, width=edge_w, color=data_color, linewidth=0,
            alpha=.25)

    # plot the relative mean absolute errors
    ax.plot(edges_mid, errors, color=err_color, marker='.',
            linewidth=2.0)
    ax.set_ylabel('Relative Mean Absolute Error', color=err_color)
    for tl in ax.get_yticklabels():
        tl.set_color(err_color)

    # plot the r2 scores
    ax2.plot(edges_mid, r2_scores, color=r2_color, linewidth=2.0, marker='.')
    ax2.set_ylabel('R2 Scores', color=r2_color)
    for tl in ax2.get_yticklabels():
        tl.set_color(r2_color)

    # tweak the viewport of axes
    ax0.set(adjustable='box-forced',
            xlim=[x_min, x_max],
            ylim=[d_min, d_max],
            yticks=[])
    ax.set(adjustable='box-forced',
           xlim=[x_min, x_max],
           ylim=[e_min, e_max],
           yticks=np.linspace(e_min, e_max, y_tick_num + 1))
    if r_max - r_min > 100.0:
        ax2_kwargs = {'yscale': 'symlog'}
    else:
        ax2_kwargs = {'yticks': np.linspace(r_min, r_max, y_tick_num + 1)}
    ax2.set(adjustable='box-forced',
            xlim=[x_min, x_max],
            ylim=[r_min, r_max],
            **ax2_kwargs)
    ax.grid()
    ax2.grid()
