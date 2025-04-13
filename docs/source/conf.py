import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "hpiPy"
author = "Reid Johnson"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
from hpipy import __version__  # noqa

version = __version__
# The full version, including alpha/beta/rc tags.
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_parser",
]

templates_path = ["_templates"]
autosummary_generate = True
exclude_patterns = []

# HTML theme settings.
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

# Output file base name for HTML help builder.
html_title = "hpiPy: House Price Indices in Python"
html_short_title = "hpiPy"

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/reidjohnson/hpipy",
            "icon": "fab fa-github fa-lg",
            "type": "fontawesome",
        }
    ],
    "show_toc_level": 2,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}

# Autosummary settings.
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
    "inherited-members": True,
}

# Intersphinx configuration.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("http://pandas.pydata.org/docs/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# MyST parser settings.
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# Copybutton settings.
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True
