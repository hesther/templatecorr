"""
short description of templatecorr.

A Python package to hierarchically correct a list of reaction templates.
"""

# Add imports here
from .extract_templates import *
from .correct_templates import *
from .helpers import *
from .canonize_templates import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
