from VLABench import *
import os
from pathlib import Path
import os
import sys
import logging

is_windows = sys.platform == 'win32'
is_linux = sys.platform.startswith('linux')
is_mac = sys.platform == 'darwin'

package_root = os.path.dirname(os.path.abspath(__file__))

env_var_name = 'VLABENCH_ROOT'


if is_windows:
    logging.info("Detect Windows, add LM4root as environment variable: ", package_root)
    os.environ.setdefault(env_var_name, package_root)
elif is_linux:
    logging.info("Detect Linux, add LM4root as environment variable: ", package_root)
    os.environ[env_var_name] =  package_root
elif is_mac:
    logging.info("Detect Linux, add LM4root as environment variable: ", package_root)
    os.environ.setdefault(env_var_name, package_root)
