# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import pathlib
import sys
import os.path

basedir = os.path.abspath(os.path.join(pathlib.Path(__file__).parents[2], "src"))
sys.path.insert(0, basedir)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'AudioTree'
copyright = '2024, David Braun'
author = 'David Braun'
first_line = open(os.path.join(pathlib.Path(__file__).parents[2], "src/audiotree/__init__.py"), "r").readline()
# first_line is '__version__ = "1.2.3"'
assert first_line.startswith("__version__ = ")
release = first_line.split("=")[1].strip()[1:-1]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'furo'
html_static_path = ['_static']
html_title = f'AudioTree documentation, v{release}'

add_module_names = False 
autoclass_signature = "separated"
todo_include_todos = True
napoleon_use_ivar = True

html_theme_options = {
    "top_of_page_buttons": [],
}