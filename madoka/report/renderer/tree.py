# -*- coding: utf-8 -*-
import io
import mimetypes
import os

from PIL import Image

from .base import ReportRenderer, ATTACHMENT_CONTENT_TYPE
from .resources import ReportResourcesManager

__all__ = [
    'ReportElementType', 'ReportElement', 'ReportBlock',
    'ReportText', 'ReportInternalLink', 'ReportImage', 'ReportAttachment',
    'ReportDataFrame', 'ReportTreeTOCNode', 'TreeReportRenderer',
]


class ReportElementType:
    """Enum for indicating the element type."""
    BLOCK = 'block'
    ROOT = 'root'
    TEXT = 'text'
    INTERNAL_LINK = 'internal_link'
    HTML = 'html'
    IMAGE = 'image'
    ATTACHMENT = 'attachment'
    DATA_FRAME = 'data_frame'


class ReportElement(object):
    """Base class for all rendered element of a report.

    Parameters
    ----------
    anchor : str
        Anchor name of this element.
        If specified, this element can be referenced by internal links.
    """
    element_type = None
    _repr_attributes = ()

    def __init__(self, anchor=None):
        self.anchor = anchor

    def __repr__(self):
        attributes = []
        for k in self._repr_attributes:
            v = getattr(self, k, None)
            if v is not None:
                v = repr(v)
                if len(v) > 20:
                    v = '%s..%s' % (v[:10], v[-8:])
                attributes.append('%s=%s' % (k, v))
        if self.anchor:
            attributes.append('anchor=%r' % (self.anchor,))
        if attributes:
            ret = '%s(%s)' % (self.__class__.__name__, ','.join(attributes))
        else:
            ret = self.__class__.__name__
        return ret


class ReportBlock(ReportElement):
    """A rendered block of a report.

    Parameters
    ----------
    title : str
        Title of this block.

    report : madoka.report.Report
        The report object, if it is a report block.

    children : collections.Iterable[ReportElement]
        Child elements of this block.

    anchor : str
        Anchor name of this element.
        If specified, this element can be referenced by internal links.
    """
    element_type = ReportElementType.BLOCK

    def __init__(self, title, report=None, children=None, anchor=None):
        super(ReportBlock, self).__init__(anchor=anchor)
        self.title = title
        self.report = report
        self.items = list(children) if children else []

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __repr__(self):
        if self.items:
            c = '\n'.join('  ' + line for line in
                          ',\n'.join(repr(i) for i in self.items).split('\n'))
            if self.title:
                ret = 'ReportBlock(title=%r,\n%s\n)' % (self.title, c)
            else:
                ret = 'ReportBlock(\n%s\n)' % c
        else:
            ret = 'ReportBlock(title=%r)' % self.title
        return ret

    def add(self, element):
        """Add a report element to this block."""
        self.items.append(element)


class ReportRoot(ReportBlock):
    """Root report block."""
    element_type = ReportElementType.ROOT

    def __init__(self, title, exp_id, children=None, anchor=None):
        super(ReportRoot, self).__init__(title=title, children=children,
                                         anchor=anchor)
        self.exp_id = exp_id


class ReportText(ReportElement):
    """A rendered text of a report.

    Parameters
    ----------
    text : str
        The report text.

    anchor : str
        Anchor name of this element.
        If specified, this element can be referenced by internal links.
    """
    element_type = ReportElementType.TEXT
    _repr_attributes = ('text',)

    def __init__(self, text, anchor=None):
        super(ReportText, self).__init__(anchor=anchor)
        self.text = text


class ReportInternalLink(ReportElement):
    """A rendered internal link of a report.

    Parameters
    ----------
    text : str
        The link text.

    target : str
        The target anchor.
    """
    element_type = ReportElementType.INTERNAL_LINK
    _repr_attributes = ('text', 'target')

    def __init__(self, text, target):
        super(ReportInternalLink, self).__init__()
        self.text = text
        self.target = target


class ReportHTML(ReportElement):
    """A rendered HTML of a report.

    Parameters
    ----------
    source : str
        The report HTML source.

    anchor : str
        Anchor name of this element.
        If specified, this element can be referenced by internal links.
    """
    element_type = ReportElementType.HTML
    _repr_attributes = ('source',)

    def __init__(self, source, anchor=None):
        super(ReportHTML, self).__init__(anchor=anchor)
        self.source = source


class _ReportFile(ReportElement):
    """A rendered file of a report.

    Parameters
    ----------
    data: bytes
        Binary content of this file.

    title : str
        Title of this file.

    filename : str
        File name of this file.

    extension : str
        Extension of this file.

    content_type : str
        Mime type of this file.

    anchor : str
        Anchor name of this element.
        If specified, this element can be referenced by internal links.
    """
    _repr_attributes = ('title', 'filename', 'extension', 'content_type')

    def __init__(self, data, title=None, filename=None, extension=None,
                 content_type=None, anchor=None):
        super(_ReportFile, self).__init__(anchor=anchor)
        if extension is None:
            if filename is not None:
                extension = os.path.splitext(filename)[1]
            else:
                extension = mimetypes.guess_extension(content_type)
                if extension is None:
                    raise RuntimeError('Unknown mime type %r.' % content_type)
        self.data = data
        self.title = title
        self.filename = filename
        self.extension = extension
        self.content_type = content_type

    @property
    def title_or_filename(self):
        return self.title or self.filename


class ReportImage(_ReportFile):
    """A rendered image of a report.

    Parameters
    ----------
    image : PIL.Image.Image | bytes | io.IOBase
        PIL image object, the content of image as bytes, or a file-like
        object that can read out the content of image.

    title : str
        Title of the image.

    content_type : str
        Content-type of the image, required if only the content of the image
        rather than a PIL image object is specified.

    anchor : str
        Anchor name of this element.
        If specified, this element can be referenced by internal links.
    """
    element_type = ReportElementType.IMAGE

    def __init__(self, image, title=None, content_type=None, anchor=None):
        ext = None
        if isinstance(image, Image.Image):
            with io.BytesIO() as f:
                image.save(f, format='PNG')
                f.seek(0)
                img = f.read()
            content_type = 'image/png'
            ext = '.png'
        elif hasattr(image, 'read'):
            img = image.read()
            if not isinstance(img, bytes):
                raise TypeError('Required to read bytes but got string.')
        elif isinstance(image, bytes):
            img = image
        else:
            raise TypeError('%r cannot be rendered as image.' % (image,))

        if content_type is None:
            raise ValueError('Content-type of the image is required.')

        super(ReportImage, self).__init__(
            img, title=title, extension=ext, content_type=content_type,
            anchor=anchor
        )


class ReportAttachment(_ReportFile):
    """A rendered attachment of a report.

    Parameters
    ----------
    data : bytes | io.IOBase
        Bytes of the attachment, or a file-like object.

    title : str
        Title of the attachment.

    content_type : str
        Content-type of the attachment.

    anchor : str
        Anchor name of this element.
        If specified, this element can be referenced by internal links.
    """
    element_type = ReportElementType.ATTACHMENT

    def __init__(self, data, filename, title=None, content_type=None,
                 anchor=None):
        if hasattr(data, 'read'):
            cnt = data.read()
            if not isinstance(cnt, bytes):
                raise TypeError('Required to read bytes but got string.')
        elif isinstance(data, bytes):
            cnt = data
        else:
            raise TypeError('%r cannot be rendered as attachment.' % (data,))

        if content_type is None:
            content_type = mimetypes.guess_type(filename)
        if content_type is None:
            content_type = ATTACHMENT_CONTENT_TYPE

        super(ReportAttachment, self).__init__(
            cnt, title=title, filename=filename, content_type=content_type,
            anchor=anchor
        )


class ReportDataFrame(ReportElement):
    """A pandas DataFrame of a report.

    Parameters
    ----------
    df : pandas.DataFrame
        Pandas data frame.

    title : str
        Title of this data frame.

    anchor : str
        Anchor name of this element.
        If specified, this element can be referenced by internal links.
    """
    element_type = ReportElementType.DATA_FRAME
    _repr_attributes = ('title',)

    def __init__(self, df, title=None, anchor=None):
        super(ReportDataFrame, self).__init__(anchor=anchor)
        self.df = df
        self.title = title


class ReportTreeTOCNode(object):
    """Node of the doc tree TOC."""

    def __init__(self, title, anchor=None, children=None):
        if children is None:
            children = []
        self.title = title
        self.anchor = anchor
        self.items = children   # type: list[ReportTreeTOCNode]

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, item):
        return self.items[item]

    def __contains__(self, item):
        return item in self.items

    @classmethod
    def from_block(cls, block):
        """Construct a new TOC node from specified block.

        Parameters
        ----------
        block : ReportBlock
            The report tree block.

        Returns
        -------
        ReportTreeTOCNode
        """
        def gather(parent, target):
            if isinstance(parent, ReportBlock):
                if parent.title:
                    items = []
                    for c in parent:
                        gather(c, items)
                    node = ReportTreeTOCNode(parent.title, parent.anchor,
                                             items)
                    target.append(node)
                else:
                    for c in parent:
                        gather(c, target)

        children = []
        for child in block:
            gather(child, children)
        return ReportTreeTOCNode(block.title, block.anchor, children)


class TreeReportRenderer(ReportRenderer):
    """Renderer that builds a doc tree for the report.

    A TreeReportRenderer builds a document tree for specified report.
    It can be used as the basis for more complicated renderer, like
    the ``HTMLReportRenderer``.

    Parameters
    ----------
    rrm : ReportResourcesManager
        Report resources manager.

    rrm_prefix : str
        Resources path prefix for the report.

    title : str
        Title of the root report block.

    exp_id : str
        Identifier of the experiment.
        If is suggested to be a path-like string with "/" as delimiter.

    log_path : str
        Path of the experiment log file.
    """

    def __init__(self, rrm, rrm_prefix='', title=None, log_path=None,
                 exp_id=None):
        super(TreeReportRenderer, self).__init__(rrm, rrm_prefix)
        self.title = title
        self.exp_id = exp_id
        self.log_path = log_path
        self.root = None   # type: ReportRoot
        self._block_stack = None

        # get the anchor of toc
        self.toc_anchor = self.get_unique_anchor(object(), name='toc')

    def get_toc(self):
        """Get the table of contents.

        Returns
        -------
        ReportTreeTOCNode
        """
        return ReportTreeTOCNode.from_block(self.root)

    def close(self):
        pass

    def begin_document(self):
        self.root = ReportRoot(title=self.title, exp_id=self.exp_id)
        self._block_stack = [self.root]
        return self

    def end_document(self):
        self._block_stack.clear()
        return self

    def _new_element(self, cls, *args, **kwargs):
        e = cls(*args, **kwargs)
        self._block_stack[-1].add(e)
        return e

    def begin_block(self, title=None, report=None):
        if title:
            anchor = self.get_unique_anchor(report, title)
        else:
            anchor = None
        block = self._new_element(ReportBlock, title=title, report=report,
                                  anchor=anchor)
        self._block_stack.append(block)
        return self

    def end_block(self):
        if self._block_stack[-1] is self.root:
            raise RuntimeError('Attempt to close the document block.')
        self._block_stack.pop()
        return self

    def write_text(self, text):
        self._new_element(ReportText, text)
        return self

    def write_html(self, html):
        self._new_element(ReportHTML, html)
        return self

    def write_image(self, image, title=None, content_type=None):
        self._new_element(ReportImage, image=image, title=title,
                          content_type=content_type)
        return self

    def write_attachment(self, data, filename, title=None, content_type=None):
        self._new_element(
            ReportAttachment, data=data, filename=filename,
            title=title, content_type=content_type
        )
        return self

    def write_data_frame(self, df, title=None):
        self._new_element(ReportDataFrame, df=df, title=title)
        return self

    def write_figure(self, fig, title=None):
        with io.BytesIO() as f:
            fig.savefig(f, format='png', dpi=90)
            f.seek(0)
            return self.write_image(f, content_type='image/png')
