# -*- coding: utf-8 -*-
import os

from jinja2 import FileSystemLoader, Environment

__all__ = []


def render_template(template_name, *args, **kwargs):
    """Render specified template."""
    return template_env.get_template(template_name).render(*args, **kwargs)


template_root = \
    os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
"""Template root directory."""

template_loader = FileSystemLoader(template_root)
"""Jinja2 template loader."""

template_env = Environment(loader=template_loader, autoescape=True)
"""Jinaj2 template environment."""
