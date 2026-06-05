import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "snn"
author = "samsit-phew"
release = "0.1.0"
copyright = "2026, samsit-phew"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

html_theme = "furo"
html_title = "snn"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "sidebar_hide_name": False,
    "light_css_variables": {
        "color-brand-primary": "#2563eb",
        "color-brand-content": "#1d4ed8",
        "color-admonition-background": "#eff6ff",
    },
    "dark_css_variables": {
        "color-brand-primary": "#60a5fa",
        "color-brand-content": "#93c5fd",
    },
    "footer_icons": [],
}

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
    "fieldlist",
]
myst_heading_anchors = 3
