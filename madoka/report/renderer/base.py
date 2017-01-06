# -*- coding: utf-8 -*-
import contextlib
import uuid

from ..helper import safe_filename

__all__ = ['ReportRenderer']

ATTACHMENT_CONTENT_TYPE = 'application/octet-stream'
"""Content-type of the attachment file."""


class ReportRenderer(object):
    """Base class for all report renderer.

    Parameters
    ----------
    rrm : ReportResourcesManager
        The report resources manager.

    rrm_prefix : str
        Resources path prefix for the report.
    """

    def __init__(self, rrm, rrm_prefix=''):
        if rrm_prefix and not rrm_prefix.endswith('/'):
            rrm_prefix += '/'
        self.rrm = rrm  # type: madoka.report.ReportResourcesManager
        self.rrm_prefix = rrm_prefix

        # dict of anchors having assigned to objects
        self._anchors = {}          # type: dict[any, str]
        self._anchor_next_id = {}   # type: dict[str, int]

    def __enter__(self):
        """Begin to render the report."""
        self.begin_document()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_document()
        self.close()

    def close(self):
        """Close all files and resources opened by this renderer."""
        raise NotImplementedError()

    def render(self, *reports):
        """Render reports with this renderer.

        Parameters
        ----------
        *reports : tuple[madoka.report.Report]
            Reports to be rendered.

        Returns
        -------
        self
        """
        for report in reports:
            report.render(self)
        return self

    def get_unique_anchor(self, obj, name=None):
        """Get a unique anchor for specified object.

        Parameters
        ----------
        obj : any
            The object which an anchor should be assigned to.

        name : str
            Suggested name of this anchor.
        """
        if obj not in self._anchors:
            if name is None:
                name = str(uuid.uuid4()).replace('-', '')
            else:
                name = safe_filename(name)
            next_id = self._anchor_next_id.get(name, 0)
            self._anchor_next_id[name] = next_id + 1
            self._anchors[obj] = '%s_%s' % (name, next_id)
        return self._anchors[obj]

    def begin_document(self):
        """Begin to render the report.

        Derived classes may override this to write report headers.
        This method will be called automatically with a context manager.

        Returns
        -------
        self
        """
        raise NotImplementedError()

    def end_document(self):
        """Finish to render the report.

        Derived classes may override this to write report endings.
        This method will be called automatically with a context manager.

        Returns
        -------
        self
        """
        raise NotImplementedError()

    def begin_block(self, title=None, report=None):
        """Begin to write a block.

        A block can be used to hint the renderer that later contents should
        be treated as a separate region from the previous contents.
        It can be used to indicate a section, a subsection, a paragraph,
        or anything else.

        Parameters
        ----------
        title : str
            The title of this block, might be used to compose contents table.

        report : Report
            The report instance, if it is a report block.

        Returns
        -------
        self
        """
        raise NotImplementedError()

    def end_block(self):
        """Finish to write a block.

        Returns
        -------
        self
        """
        raise NotImplementedError()

    @contextlib.contextmanager
    def block(self, title=None):
        """Open a block context."""
        try:
            self.begin_block(title=title)
            yield
        finally:
            self.end_block()

    def write_text(self, text):
        """Write a piece of text.

        The text should be interpreted as plain text, and should be rendered
        in the most suitable way, such that the paragraphs and indents are
        preserved in resulted document.

        Parameters
        ----------
        text : str
            The plain text.

        Returns
        -------
        self
        """
        raise NotImplementedError()

    def write_html(self, html):
        """Write a piece of HTML source code.

        The renderer should try its best to display the HTML code in the
        most suitable way.

        Parameters
        ----------
        html : str
            The html source code to be displayed.

        Returns
        -------
        self
        """
        raise NotImplementedError()

    def write_image(self, image, title=None, content_type=None):
        """Write an image.

        Parameters
        ----------
        image : PIL.Image | io.RawIOBase | bytes
            The image to be displayed.

        content_type : str
            Content type of the image, if the image binary contents instead
            of a PIL image instance is provided.

        title : str
            Optional title of the image.

        Returns
        -------
        self
        """
        raise NotImplementedError()

    def write_attachment(self, data, filename, title=None, content_type=None):
        """Write an attachment.

        The attachment may be rendered as a link that allow the user to
        download the file through browser.

        Parameters
        ----------
        data : io.RawIOBase | bytes
            Binary content or input stream of this attachment.

        filename : str
            File name of this attachment.

        title : str
            Alternative title of this attachment, other than the filename.

        content_type : str
            Content type of this attachment.

        Returns
        -------
        self
        """
        raise NotImplementedError()

    def write_data_frame(self, df, title=None):
        """Write a data frame, as a table.

        Parameters
        ----------
        df : pandas.DataFrame
            The data frame to be displayed.

        title : str
            Optional title of the data frame.

        Returns
        -------
        self
        """
        raise NotImplementedError()

    def write_figure(self, fig, title=None):
        """Write a matplotlib figure.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to be displayed.

        title : str
            Optional title of the figure.

        Returns
        -------
        self
        """
        raise NotImplementedError()
