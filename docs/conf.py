import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = "mem0rylol"
copyright = "2023, toeknee"
author = "toeknee"
version = "0.2"
release = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_static_path = ["_static"]