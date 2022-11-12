#!/usr/bin/env python3
import os, sys
from pathlib import Path
sys.path.append(os.getcwd())

import FisInMa


# Generate plots for the discretization
from source.getting_started import plot_discretization
from source.documentation import plot_penalty_functions

plot_discretization.plot_default_discretization(outdir=Path(os.path.dirname(plot_discretization.__file__)))
plot_penalty_functions.plot_discretization_product(outdir=Path(os.path.dirname(plot_penalty_functions.__file__)))
plot_penalty_functions.plot_discrete_penalty_individual_template(outdir=Path(os.path.dirname(plot_penalty_functions.__file__)))