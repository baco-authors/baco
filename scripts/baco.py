import os
import sys
import warnings
from collections import OrderedDict

sys.path.append(".")  # noqa
from baco import baco


if len(sys.argv) == 2:
    parameters_file = sys.argv[1]
else:
    print("Error: only one argument needed, the parameters json file.")

baco.optimize(parameters_file)
