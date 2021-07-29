"""
Unit and regression test for the templatecorr package.
"""

# Import package, test suite, and other packages as needed
import templatecorr
import pytest
import sys

def test_templatecorr_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "templatecorr" in sys.modules
