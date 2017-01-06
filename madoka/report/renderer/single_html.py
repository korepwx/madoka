# -*- coding: utf-8 -*-
import base64
import io
import warnings

import six
from markupsafe import escape

from .base import ReportRenderer
from .resources import ReportResourcesManager
from .tree import ReportImage, ReportAttachment

__all__ = ['SingleHTMLReportRenderer']


class SingleHTMLReportRenderer(ReportRenderer):
    """Single page report renderer.

    This class will render the reports as a single HTML file.
    The resources will be embedded in the HTML file, via base64 encoding.

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
        super(SingleHTMLReportRenderer, self).__init__(rrm, rrm_prefix)
        self.f = io.BufferedWriter(file)
        self.title = title
        self.exp_id = exp_id
        self.log_path = log_path
        self.close_file = close_file
        self._block_level = 0

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
        SingleHTMLReportRenderer
        """
        return SingleHTMLReportRenderer(
            open(path, 'wb'), title, exp_id, log_path, rrm, rrm_prefix,
            close_file=True
        )

    def close(self):
        if self.close_file:
            self.f.close()
        self.f = None

    def _write_pieces(self, *pieces):
        """Write string pieces."""
        for p in pieces:
            if isinstance(p, six.text_type):
                self.f.write(p.encode('utf-8'))
            else:
                self.f.write(p)
        return self

    def begin_document(self):
        return self._write_pieces(
            b'<html lang="en"><head><meta http-equiv="Content-Type" '
            b'content="text/html; charset=utf-8" /><title>',
            escape(self.title),
            b'</title></head><body>'
        )

    def end_document(self):
        if self._block_level != 0:
            warnings.warn('Some block is not closed properly.')
        self._write_pieces(b'</body></html>')
        return self

    def begin_block(self, title=None, report=None):
        self._block_level += 1
        if title is not None:
            self._write_pieces(
                b'<h%d>' % self._block_level,
                escape(title),
                b'</h%d><div class="named_block">' % self._block_level
            )
        else:
            self._write_pieces(b'<div class="unnamed_block">')
        return self

    def end_block(self):
        if self._block_level <= 0:
            raise RuntimeError('Attempt to close the document block.')
        self._block_level -= 1
        return self._write_pieces(b'</div>')

    def write_text(self, text):
        if isinstance(text, six.text_type):
            _r, _n, _w, _we, _br = '\r', '\n', ' ', '&nbsp;', '<br/>'
        else:
            _r, _n, _w, _we, _br = b'\r', b'\n', b' ', b'&nbsp;', b'<br/>'
        lines = filter(
            lambda s: s,
            (line.rstrip(_r) for line in text.split(_n))
        )
        lines = (escape(i) for i in lines)
        lines = (i.replace(_w, _we) for i in lines)
        return self._write_pieces((_br + _br).join(lines))

    def write_html(self, html):
        return self._write_pieces(html)

    def write_image(self, image, title=None, content_type=None):
        img = ReportImage(image, title=title,
                          content_type=content_type)
        self._write_pieces(
            b'<img style="max-width:600px" src="data:', img.content_type,
            b';base64,', base64.b64encode(img.data), b'"')
        if img.title is not None:
            self._write_pieces(b' title="', escape(img.title), b'"')
        return self._write_pieces(b' />')

    def write_attachment(self, data, filename, title=None, content_type=None):
        attach = ReportAttachment(data, filename, title=title,
                                  content_type=content_type)
        return self._write_pieces(
            b'Attachment: <a href="data:', attach.content_type, b';base64,',
            base64.b64encode(attach.data), b'" download="',
            escape(attach.filename), b'">',
            escape(attach.title or attach.filename), b'</a>')

    def write_data_frame(self, df, title=None):
        self._write_pieces(b'<div class="dataframe">')
        if title is not None:
            self._write_pieces(
                b'<div class="title">',
                escape(title),
                b'</div>'
            )
        return self._write_pieces(df.to_html(), b'</div>')

    def write_figure(self, fig, title=None):
        with io.BytesIO() as f:
            fig.savefig(f, format='png', dpi=90)
            f.seek(0)
            self.write_image(f, content_type='image/png')
        return self
