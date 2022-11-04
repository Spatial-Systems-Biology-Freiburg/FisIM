#!/usr/bin/env python3
import os, sys
from pathlib import Path
sys.path.append(os.getcwd())

import FisInMa


# Generate plots for the discretization
from source.user_interface import plot_discretization
plot_discretization.plot_default_discretization(outdir=Path(os.path.dirname(plot_discretization.__file__)))
