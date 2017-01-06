# -*- coding: utf-8 -*-
import gzip
import io
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from madoka.utils import wrap_text_writer
from .base import Report
from .helper import classfication_report_data_frame

__all__ = [
    'classification_report', 'ClassificationReport', 'BinaryClassifyReport',
]


def classification_report(truth, predict, proba, name='Classification Report'):
    """Create a classification report.

    Parameters
    ----------
    truth : numpy.ndarray
        Ground truth (correct) target values.

    predict : numpy.ndarray
        Estimated target as returned by the classifier.

    proba : numpy.ndarray
        Estimated probabilities for each target to be each class.

    name : str
        Name of this report.
    """
    if len(proba.shape) == 1 or proba.shape[1] <= 2:
        return _BinaryClassificationReport(truth, predict, proba, name=name)
    return ClassificationReport(truth, predict, proba, name=name)


class ClassificationReport(Report):
    """Classification report."""

    _ATTRIBUTE_LEGACY_MAPPING = {
        'y_true': 'truth', 'y_pred': 'predict', 'y_proba': 'proba'
    }
    _PREDICT_FN = 'predict.csv.gz'

    def __init__(self, truth, predict, proba, name=None):
        super(ClassificationReport, self).__init__(name=name)
        self.truth = truth
        self.predict = predict
        self.proba = proba

    def _make_compressed_csv(self):
        if getattr(self, '_compressed_csv', None) is None:
            with io.BytesIO() as f:
                # Note we force `mtime=0` here.
                #
                # This is because we need the compressed gz to be identical
                # for the same CSV files, in order to use report resources
                # manager.
                with gzip.GzipFile(fileobj=f, mode='wb', mtime=0) as out:
                    # generate the probability columns
                    y_proba = self.proba
                    if len(y_proba.shape) == 2 and y_proba.shape[1] == 1:
                        y_proba = y_proba.reshape([-1])
                    if len(y_proba.shape) == 1:
                        proba_cols = [('p(class=1)', y_proba)]
                    else:
                        proba_cols = [('p(class=%d)' % i, y_proba[:, i])
                                      for i in range(y_proba.shape[1])]

                    # generate the data frame
                    data = {'predict': self.predict, 'truth': self.truth}
                    for c, v in proba_cols:
                        data[c] = v
                    columns = ['predict', 'truth'] + [c for c, _ in proba_cols]
                    csv_df = pd.DataFrame(data=data, columns=columns)
                    writer = wrap_text_writer(out, encoding='utf-8')
                    csv_df.to_csv(writer, index=False, header=True)
                f.seek(0)
                self._compressed_csv = f.read()
        return self._compressed_csv

    def _render_content(self, renderer):
        with renderer.block('Classification Metrics'):
            with renderer.block():
                df = classfication_report_data_frame(self.truth, self.predict)
                renderer.write_data_frame(df)
            with renderer.block():
                renderer.write_attachment(self._make_compressed_csv(),
                                          filename=self._PREDICT_FN,
                                          content_type='text/csv+gzip')


class _BinaryClassificationReport(ClassificationReport):
    """Classification report dedicated for binary classifiers."""

    def __init__(self, truth, predict, proba, name=None):
        if len(proba.shape) == 2:
            if proba.shape[1] == 1:
                proba = proba.reshape([-1])
            elif proba.shape[1] == 2:
                proba = proba[:, 1]
            else:
                raise TypeError('The output probability is not binary.')
        super(_BinaryClassificationReport, self).__init__(
            truth, predict, proba, name=name)

    def _make_figure(self):
        if getattr(self, '_figure', None) is None:
            truth = self.truth
            proba = self.proba

            def plot_sub(ax, label, color, pos=1, lw=2):
                if pos == 1:
                    p, r, th = precision_recall_curve(truth, proba)
                    area = average_precision_score(truth, proba)
                else:
                    a, b = 1 - truth, 1.0 - proba
                    p, r, th = precision_recall_curve(a, b)
                    area = average_precision_score(a, b)
                ax.plot(r, p, lw=lw, color=color,
                        label='Precision-Recall curve of %s (area=%.4f)' %
                              (label, area))

            from matplotlib import pyplot as plt
            fig = plt.figure(figsize=(12, 9))
            ax = plt.subplot(111)
            plot_sub(ax, 'class 0', pos=0, color='navy')
            plot_sub(ax, 'class 1', pos=1, color='orangered')

            # tweak figure styles
            ax.grid()
            ax.set_title('Precision-Recall Curve of Binary Classification',
                         y=1.15)
            ax.set_xlim([0.0, 1.0])
            ax.set_xticks(np.linspace(0.0, 1.0, 11))
            ax.set_ylim([0.0, 1.05])
            ax.set_yticks(np.linspace(0.0, 1.0, 11))
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')

            # Put a legend above current axis
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width, box.height * 0.9])
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1)
            self._figure = fig
        return self._figure

    def _render_content(self, renderer):
        super(_BinaryClassificationReport, self)._render_content(renderer)
        with renderer.block('Precision-Recall Curve'):
            renderer.write_figure(self._make_figure())


class BinaryClassifyReport(_BinaryClassificationReport):

    def __init__(self, truth, predict, proba, name=None):
        warnings.warn(
            'Creating instance of `BinaryClassifyReport` has been deprecated. '
            'Please use `madoka.report.classification_report` function to '
            'create a report object.',
            category=DeprecationWarning
        )
        super(BinaryClassifyReport, self).__init__(
            truth, predict, proba, name=name)
