# -*- coding: utf-8 -*-
import os

import numpy as np

from .base import Report, ReportGroup

__all__ = ['tf_summary_report', 'TFSummaryReport', 'TFSummaryReportGroup']


def tf_summary_report(summary_dir, name='Training Summary'):
    """Create a TensorFlow training summary report from given summary dir.

    Parameters
    ----------
    summary_dir : str
        Path of the summary directory.

    name : str
        Name of the report.

    Returns
    -------
    TFSummaryReport | TFSummaryReportGroup
        If the specified directory contains only one summary, then
        returns TFSummaryReport.  Otherwise returns a TFSummaryReportGroup.
    """
    from madoka.utils import find_tf_summary_dirs, read_tf_summary_dir
    summary_dirs = [p for p, _ in find_tf_summary_dirs(summary_dir)]

    if len(summary_dirs) >= 2:
        # if there's more than one summary directory,
        # construct the summary report group, with the relative path
        # as the name of each summary report.
        children = []
        for p in summary_dirs:
            relpath = os.path.relpath(p, summary_dir)
            n = '/' if relpath == '.' else '/' + relpath
            children.append(
                TFSummaryReport(
                    read_tf_summary_dir(p),
                    name=n
                )
            )
        report = TFSummaryReportGroup(children=children, name=name)
    elif len(summary_dirs) == 1:
        # if there's only one summary directory,
        # just make an individual summary report
        report = TFSummaryReport(
            read_tf_summary_dir(summary_dirs[0]),
            name=name
        )
    else:
        # if there's no summary, then construct an empty group.
        report = TFSummaryReportGroup(children=[], name=name)

    return report


class TFSummaryReport(Report):
    """TensorFlow training summary report.

    Parameters
    ----------
    summary : madoka.utils.tfhelper.TensorFlowSummary
        Training summary object.

    name : str
        Name of this report.
    """

    def __init__(self, summary, name=None):
        super(TFSummaryReport, self).__init__(name)
        self.summary = summary

        # lazy initialized members
        self._figure = None

    def _make_loss_figure(self):
        if getattr(self, '_figure', None) is None:
            from matplotlib import pyplot as plt

            # if the summary is empty, we should return no figure
            s = self.summary
            if s.training_loss is None and s.validation_loss is None:
                return None

            # plot the figure
            fig = plt.figure(figsize=(12, 9))
            ax = plt.subplot(111)
            scalars = (
                ('training_loss', ('training', 'orangered')),
                ('validation_loss', ('validation', 'navy')),
            )

            def plot_scalar(ax, x, y, tag, color, lw=1, alpha=1.):
                ax.plot(x, y, lw=lw, color=color, label=tag, alpha=alpha)

            y_max = 0
            x_max = 0
            # plot the full data
            for tag, (label, color) in scalars:
                series = getattr(s, tag, None)
                if series is not None:
                    x, y = series.index, series.data
                    plot_scalar(ax, x, y, label, color, lw=1, alpha=.2)
                    x_max = max(x_max, np.max(series.index.data))
                    y_max = max(y_max,
                                np.mean(series.data) + 3. * np.std(series.data))

            # plot the smoothed data
            for tag, (label, color) in scalars:
                series = getattr(s, tag, None)
                if series is not None:
                    w = int(len(series) * 0.01)
                    if w > 1:
                        series = series.rolling(window=w).mean()
                    x, y = series.index, series.data
                    label = '%s (smooth)' % label
                    plot_scalar(ax, x, y, label, color, lw=2, alpha=.9)

            # tweak figure styles
            ax.grid()
            ax.set_title('Training and validation loss', y=1.15)
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.set_xlim([0, x_max])
            ax.set_ylim([0.0, y_max])
            ax.set_yticks(np.linspace(0.0, y_max, 11))

            # Put a legend above current axis
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width, box.height * 0.9])
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)

            self._figure = fig
        return self._figure

    def _render_content(self, renderer):
        fig = self._make_loss_figure()
        if fig is not None:
            renderer.write_figure(self._make_loss_figure())


class TFSummaryReportGroup(ReportGroup):
    """TensorFlow training summary report group.

    A TrainSummaryReportGroup is usually constructed from a summary directory,
    so as to contain all the training summaries in that directory.

    Parameters
    ----------
    children : collections.Iterable[TFSummaryReport]
        The child summary reports.

    name : str
        Name of this report.
    """

    def __init__(self, children, name=None):
        super(TFSummaryReportGroup, self).__init__(children, name)
