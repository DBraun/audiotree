# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import pathlib
import sys

basedir = (pathlib.Path(__file__).parents[2] / "src").resolve()
sys.path.insert(0, str(basedir))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "AudioTree"
copyright = "2024-%Y, David Braun"  # Sphinx substitutes %Y with the build year.
author = "David Braun"

# Read the version from the package itself (importable via the sys.path insert
# above), the same source pyproject.toml uses (`attr = "audiotree.__version__"`).
import audiotree  # noqa: E402

version = release = audiotree.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

# Resolve :func:`flax.nnx.scan`-style cross-references against these projects'
# own documentation inventories.
intersphinx_mapping = {
    "flax": ("https://flax.readthedocs.io/en/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "grain": ("https://google-grain.readthedocs.io/en/latest/", None),
}

# Execute the ``>>>`` / ``.. testcode::`` examples at ``make doctest`` so the docs stay correct.
# These names are pre-imported so each example can stay concise.
doctest_global_setup = """
# grain 0.2.17+ reads absl flags in its multiprocessing prefetch; the doctest
# runner is not an ``absl.app`` entry point, so parse them with defaults.
from absl import flags
flags.FLAGS.mark_as_parsed()

import jax
import jax.numpy as jnp
import numpy as np
from audiotree import AudioTree, AudioWriter, TreeWriter
from audiotree.sources import (
    AudioDataSource,
    TreeDataSource,
    create_audio_dataset,
    create_balanced_audio_dataset,
)
"""

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = "furo"
html_static_path = ["_static"]
html_title = f"AudioTree documentation, v{release}"

add_module_names = False
autoclass_signature = "separated"
todo_include_todos = True
napoleon_use_ivar = True

html_theme_options = {
    "top_of_page_buttons": [],
}
