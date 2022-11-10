import pprint
import shutil

from FisInMa.model import FisherModelParametrized, FisherResults
from FisInMa.solving import display_fsmp_details, display_fsr_details, display_heading


def display_optimization_start(fsmp: FisherModelParametrized):
    # Create the pretty-printer
    terminal_size = shutil.get_terminal_size((80, 20))
    display_heading("SUMMARY OF FISHER MODEL")

    pp = pprint.PrettyPrinter(indent=2, width=terminal_size[0])

    display_fsmp_details(fsmp, pp)
    display_heading("STARTING OPTIMIZATION RUN")


def display_optimization_end(fsr: FisherResults):
    terminal_size = shutil.get_terminal_size((80, 20))
    print()
    pp = pprint.PrettyPrinter(indent=2, width=terminal_size[0])
    display_fsr_details(fsr, pp)
