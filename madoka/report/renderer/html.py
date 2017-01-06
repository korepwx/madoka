# -*- coding: utf-8 -*-
import io
import os

from .resources import ReportResourcesManager
from .template_utils import render_template, template_root
from .tree import TreeReportRenderer, _ReportFile

__all__ = ['HTMLReportRenderer']


class HTMLReportRenderer(TreeReportRenderer):
    """HTML report renderer.

    This renderer will save asset files to the resource manager, and render
    an HTML page which links to these resources.

    Parameters
    ----------
    file
        Target file where the reported should be rendered into.

    title : str
        Title of the HTML page.

    exp_id : str
        Identifier of the experiment.
        If is suggested to be a path-like string with "/" as delimiter.

    log_path : str
        Path of the experiment log file.

    rrm : ReportResourcesManager
        The report resources manager.
        Will not be closed by this renderer.

    rrm_prefix : str
        Resources path prefix for the report.

    close_file : bool
        Whether or not to close the given file after rendering?
    """

    def __init__(self, file, title, exp_id, log_path, rrm, rrm_prefix='',
                 close_file=False):
        super(HTMLReportRenderer, self).__init__(
            rrm, rrm_prefix, log_path=log_path, title=title, exp_id=exp_id)
        self.f = io.BufferedWriter(file)
        self.close_file = close_file

        # lazy initialized members
        self._saved_files = {}

    @classmethod
    def open(cls, path, title='Experiment Report', exp_id=None, log_path=None,
             rrm=None, rrm_prefix='resources'):
        """Render reports into specified file.

        Parameters
        ----------
        path : str
            Path of the HTML file.

        title : str
            Title of the HTML page.

        exp_id : str
            Identifier of the experiment.
            If is suggested to be a path-like string with "/" as delimiter.

        log_path : str
            Path of the experiment log file.

        rrm : ReportResourcesManager
            The report resources manager.

            If not specified, will open a new resources manager at
            `dirname(path) + rrm_prefix`.

        rrm_prefix : str
            Resources path prefix for the report.

        Returns
        -------
        HTMLReportRenderer
        """
        if rrm is None:
            rrm_path = os.path.join(os.path.dirname(path), rrm_prefix)
            rrm = ReportResourcesManager(rrm_path)
        return HTMLReportRenderer(open(path, 'wb'), title, exp_id, log_path,
                                  rrm, rrm_prefix, close_file=True)

    def save_file(self, f):
        """Save resource file to the report resources manager.

        Parameters
        ----------
        f : _ReportFile | str
            Report file element, or path of a static file.

        Returns
        -------
        str
            The filename of this file, relative to resource root.
        """
        if f not in self._saved_files:
            if isinstance(f, _ReportFile):
                ret = self.rrm.save_bytes(f.data, f.extension, f.filename)
            else:
                path = os.path.abspath(os.path.join(template_root, f))
                if not path.startswith(template_root):
                    raise ValueError('%r is not under template directory!' % f)
                ret = self.rrm.save_file(
                    path, filename=os.path.relpath(path, template_root))
            self._saved_files[f] = self.rrm_prefix + ret
        return self._saved_files[f]

    def close(self):
        if self.close_file:
            self.f.close()
        self.f = None

    def end_document(self):
        page = render_template(
            'html/index.html', root=self.root, save_file=self.save_file,
            log_path=self.log_path, toc=self.get_toc(),
            toc_anchor=self.toc_anchor
        )
        self.f.write(page.encode('utf-8'))
        return super(HTMLReportRenderer, self).end_document()
