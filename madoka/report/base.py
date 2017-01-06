# -*- coding: utf-8 -*-
import six

__all__ = ['Report', 'ReportGroup']


class Report(object):
    """Base class for all report objects.

    A report object contains a piece of evaluation result.  Such objects can
    be further grouped in order to form a richer report.

    Parameters
    ----------
    name : str
        Name of this report.

        If specified, this report would typically be rendered as a individual
        section, and this name should be the title of the section.
    """

    # Attribute legacy mappings, for loading old report objects
    _ATTRIBUTE_LEGACY_MAPPING = {}

    def __init__(self, name=None):
        self.name = name

    def __getstate__(self):
        return {
            k: v for k, v in six.iteritems(self.__dict__)
            if not k.startswith('_')
        }

    def __setstate__(self, state):
        processed = {}

        # deal with attribute legacy mapping
        for k, v in six.iteritems(state):
            if k in self._ATTRIBUTE_LEGACY_MAPPING:
                k = self._ATTRIBUTE_LEGACY_MAPPING[k]
            if not k.startswith('_'):
                processed[k] = v

        # restore attributes not started with "_"
        self.__dict__.update(processed)

        # clear attributes started with "_"
        for k in self.__dict__:
            if k.startswith('_'):
                self.__dict__[k] = None

    def render(self, renderer):
        """Render this report object via specified renderer.

        Parameters
        ----------
        renderer : madoka.report.ReportRenderer
            The report renderer.
        """
        renderer.begin_block(self.name, self)
        self._render_content(renderer)
        renderer.end_block()

    def _render_content(self, renderer):
        """Derived classes should override this to actual render the report."""
        raise NotImplementedError()


class ReportGroup(Report):
    """Gather reports into group.

    If the group itself as well as its children both have names, then the
    children would become subsections of the group section.

    Parameters
    ----------
    children : collections.Iterable[Report]
        Children of this report group.

    name : str
        Name of this report group.
    """
    # we once used `_children` attribute, however, we now have the restriction
    # that Report classes should keep persistent values in public attributes.
    _ATTRIBUTE_LEGACY_MAPPING = {'_children': 'children'}

    def __init__(self, children=None, name=None):
        super(ReportGroup, self).__init__(name=name)
        if children is not None:
            children = list(children)
        else:
            children = []
        self.children = children    # type: list[Report]

    def __iter__(self):
        return iter(self.children)

    def __len__(self):
        return len(self.children)

    def __getitem__(self, item):
        return self.children[item]

    def add(self, report):
        """Add a report into this group."""
        self.children.append(report)

    def remove(self, report):
        """Report a report from this group."""
        if report in self.children:
            self.children.remove(report)

    def _render_content(self, renderer):
        for c in self.children:
            c.render(renderer)
