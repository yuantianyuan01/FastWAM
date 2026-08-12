# XPolicyLab.py
# Make this module behave like a package so `XPolicyLab.<submodule>` works.

import os as _os

# Tell Python to search submodules under this directory
__path__ = [_os.path.dirname(__file__)]