"""Central configuration for the solar active region pipeline.

All scripts import from this module rather than defining their own
paths, thresholds, or instrument settings. Add new variables here as
each script is refactored, grouped by the section they belong to.

Machine-specific values (paths, credentials) live in local_settings.py,
which is gitignored — see local_settings_example.py for the template.
"""

import os

try:
    from local_settings import MASTER_DIR, JSOC_REGISTERED_EMAIL
except ImportError as exc:
    raise ImportError(
        'local_settings.py not found. Copy local_settings.example.py to '
        'local_settings.py and fill in your machine-specific paths and '
        'credentials.'
    ) from exc

# --------------------------------------------------------------------------
# General
# --------------------------------------------------------------------------
# Number of most recent time-steps to download per HARP (each time-step
# yields 3 files: continuum, Br, bitmap).
N_RECENT_TIMESTEPS = 2

# --------------------------------------------------------------------------
# HMI download settings
# --------------------------------------------------------------------------
# Near-real-time lookback window, in hours, ending at the time the script
# is run. Also defines the window used to scan for currently available
# NRT HARPs — in practice this will be much smaller than 24.
N_HOURS = 1

MAX_RETRIES = 10
RETRY_WAIT_SECONDS = 15
FAILURE_WAIT_SECONDS = 30

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
DOWNLOAD_DIR = MASTER_DIR + '{file}'
DOWNLOAD_LOG_FILE = os.path.join(MASTER_DIR, 'download.log')
SCAN_DIR = MASTER_DIR
