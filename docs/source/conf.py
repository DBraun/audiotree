# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'AudioTree'
copyright = '2024, David Braun'
author = 'David Braun'
release = '0.0.5'

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
html_title = 'AudioTree documentation'

add_module_names = False 
autoclass_signature = "separated"
todo_include_todos = True
napoleon_use_ivar = True

html_theme_options = {
    "top_of_page_buttons": [],
}

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import pathlib
import sys
import os.path

basedir = os.path.abspath(os.path.join(pathlib.Path(__file__).parents[2], "src"))
sys.path.insert(0, basedir)