import pprint
import shutil

from eDPM.model import FisherResults
from eDPM.solving import display_heading, display_entries, generate_matrix_cols


def display_identifiability_check(check):
    terminal_size = shutil.get_terminal_size((80, 20))
    print()
    pp = pprint.PrettyPrinter(indent=2, width=terminal_size[0])
    display_analysis_details(check, pp=pp)


def display_analysis_details(check, pp=pprint.PrettyPrinter(indent=2, width=shutil.get_terminal_size((80, 20))[0]), terminal_size=shutil.get_terminal_size((80, 20))):
    display_heading("ANALYSIS RESULTS")
    cols = generate_matrix_cols(check, "structural identifiability", terminal_size)
    display_entries(cols, terminal_size)