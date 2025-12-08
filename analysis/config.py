"""
Backward compatibility wrapper for config.py

config.py has been moved to analysis/utils/config.py
This file provides backward compatibility for existing imports.
"""

from analysis.utils.config import *

# Backward compatibility: RESULTS_PLOTS_DIR points to RESULTS_FIGURES_DIR
RESULTS_PLOTS_DIR = RESULTS_FIGURES_DIR

