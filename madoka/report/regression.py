# -*- coding: utf-8 -*-
import io
import zipfile

import numpy as np
import pandas as pd

from madoka.utils import wrap_text_writer
from .base import Report, ReportGroup
from .helper import (regression_report_data_frame,
                     normalize_regression_report_args,
                     safe_filename)

__all__ = ['regression_report', 'RegressionReport']


def regression_report(truth, predict, label=None, targets=None,
                      per_target_summary=True, name=None):
    """Create a regression report.

    Parameters
    ----------
    truth : numpy.ndarray
        Ground truth (correct) target values.

    predict : numpy.ndarray
        Estimated target as returned by a regressor.

    label : numpy.ndarray
        Label for each testing data, as an auxiliary information to
        evaluate the regression result. (Optional)

    targets : numpy.ndarray
        Names of targets, as an auxiliary information to evaluate
        the regression result. (Optional)

    per_target_summary : bool
        If True, will generate per-target regression summary.
        Otherwise will only generate the total regression summary.

    name : str
        Name of this report.
    """
    return RegressionReport(truth, predict, label, targets,
                            per_target_summary=per_target_summary,
                            name=name)


class RegressionReport(ReportGroup):
    """Generic regression report."""

    _ATTRIBUTE_LEGACY_MAPPING = {
        'features': 'targets',
        'per_feature_summary': 'per_target_summary',
    }

    def __init__(self, truth, predict, label=None, targets=None,
                 per_target_summary=True, name=None):
        assert (predict.shape == truth.shape)
        self.truth = truth
        self.predict = predict
        self.label = label
        self.targets = targets
        self.per_target_summary = per_target_summary
        super(RegressionReport, self).__init__(
            children=self._make_child_reports(),
            name=name
        )

    def _make_child_reports(self):
        # some old reports do not have `per_target_summary` attribute
        per_target_summary = getattr(self, 'per_target_summary', True)

        truth, predict, label, targets = normalize_regression_report_args(
            self.truth, self.predict, self.label,
            # some old reports do not have `targets` attribute
            getattr(self, 'targets', None),
            per_target=True
        )
        children = [
            _RegressionAllDimReport(
                truth, predict, label, targets,
                per_target_summary=per_target_summary,
                name='Regression Summary'
            )
        ]
        return children

    def _render_content(self, renderer):
        # Some old regression reports are not designed to be a ReportGroup.
        # In order to be compatible to these reports, we create the child
        # reports is the children attribute is not initialized.
        if getattr(self, 'children', None) is None:
            self.children = self._make_child_reports()
        return super(RegressionReport, self)._render_content(renderer)


class _RegressionAllDimReport(Report):
    """Full dimensional regression report.

    This class implements a regression report for all dimensions.
    It generates a table, which contains the regression metrics
    for each of the dimensions, as well as all dimensions.
    """

    _REGRESSION_ZIP_FN = 'regression.zip'
    _ATTRIBUTE_LEGACY_MAPPING = {
        'features': 'targets',
        'per_feature_summary': 'per_target_summary',
    }

    def __init__(self, truth, predict, label=None, targets=None,
                 per_target_summary=True, name=None):
        assert (predict.shape == truth.shape)
        super(_RegressionAllDimReport, self).__init__(name=name)
        self.truth = truth
        self.predict = predict
        self.label = label
        self.targets = targets
        self.per_target_summary = per_target_summary

    def _make_regression_zip(self, summary_df):
        def make_predict_csv(truth, predict):
            df = pd.DataFrame(data=np.asarray([truth, predict]).T,
                              columns=['truth', 'predict'])
            with io.BytesIO() as f:
                writer = wrap_text_writer(f, encoding='utf-8')
                df.to_csv(writer, index=False, header=True)
                f.seek(0)
                return f.read()

        def make_summary_csv():
            with io.BytesIO() as f:
                writer = wrap_text_writer(f, encoding='utf-8')
                summary_df.to_csv(writer, index=True, header=True)
                f.seek(0)
                return f.read()

        def make_label_csv(label):
            df = pd.DataFrame(data=label, columns=['label'])
            with io.BytesIO() as f:
                writer = wrap_text_writer(f, encoding='utf-8')
                df.to_csv(writer, index=False, header=True)
                f.seek(0)
                return f.read()

        if getattr(self, '_predict_zip', None) is None:
            truth, predict, label, targets = normalize_regression_report_args(
                self.truth, self.predict, self.label, self.targets,
                per_target=True
            )

            with io.BytesIO() as buf:
                with zipfile.ZipFile(
                        buf, 'w', compression=zipfile.ZIP_DEFLATED) as out:
                    # generate summary csv
                    out.writestr('summary.csv', make_summary_csv())

                    # generate label csv if necessary
                    if self.label is not None:
                        out.writestr('label.csv', make_label_csv(self.label))

                    # generate predict csv
                    for i, feat in enumerate(self.targets):
                        out.writestr(
                            '%d-%s.csv' % (i, safe_filename(feat)),
                            make_predict_csv(truth[:, i], predict[:, i])
                        )
                buf.seek(0)
                self._predict_zip = buf.read()
        return self._predict_zip

    def _render_content(self, renderer):
        with renderer.block():
            df = regression_report_data_frame(
                self.truth, self.predict, self.label, self.targets,
                self.per_target_summary
            )
            renderer.write_data_frame(df)
        with renderer.block():
            renderer.write_attachment(self._make_regression_zip(df),
                                      filename=self._REGRESSION_ZIP_FN,
                                      content_type='application/zip')
