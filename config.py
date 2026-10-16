"""
Central configuration for the solar active region complexity pipeline.

Machine-specific values (paths, credentials) should be in local_settings.py;
see local_settings_example.py for a template.
"""

import os

try:
    from local_settings import HMI_DIR, JSOC_REGISTERED_EMAIL, GONG_DIR
except ImportError as exc:
    raise ImportError(
        'local_settings.py not set up correctly. Copy local_settings.example.py to '
        'local_settings.py and fill in your machine-specific paths and credentials.'
    ) from exc

# --------------------------------------------------------------------------
# download_hmi.py settings
# --------------------------------------------------------------------------
# Near-real-time lookback window, in hours, ending at the time the script
N_HOURS = 1.5

MAX_RETRIES = 10
RETRY_WAIT_SECONDS = 15
FAILURE_WAIT_SECONDS = 30

DOWNLOAD_DIR = HMI_DIR + '{file}'
DOWNLOAD_LOG_FILE = os.path.join(HMI_DIR, 'download.log')
SCAN_DIR = HMI_DIR



# --------------------------------------------------------------------------
# download_gong.py settings
# --------------------------------------------------------------------------
# Delete the pre-existing files before attempting to download new ones?
GONG_CLEAN_BEFORE_DOWNLOAD = True

GONG_BASE_URL = 'https://gong2.nso.edu/oQR'
GONG_MAG_PRODUCT = 'bqa'
GONG_CON_PRODUCT = 'iqa'
 
# The six active GONG network sites. 'tc'/'te' also appear in GONG's archive
# directories but are Boulder, CO engineering/testbed sites, not part of
# the operational network — do not add them here. NOTE: Mauna Loa ('ml') has
# had no data since being shut down following a 2022 volcanic eruption; left in
# here so it resumes automatically if the site comes back online.
GONG_SITES = {
    'bigbear': 'bb',
    'mauna_loa': 'ml',
    'learmonth': 'le',
    'udaipur': 'ud',
    'el_teide': 'td',
    'cerro_tololo': 'ct',
}
 
# Number of most-recent continuum timestamps to try per site before
# giving up on that site, regardless of whether a given try fails due to
# no matching magnetogram timestamp or a failed VALID check.
GONG_MAX_CANDIDATES = 3
 
# How many days back to search per site (today, plus this many prior
# days) — covers the case where today's directory is empty or has fewer
# than GONG_MAX_CANDIDATES entries so far.
GONG_LOOKBACK_DAYS = 2
 
GONG_DOWNLOAD_LOG_FILE = os.path.join(GONG_DIR, 'download.log')



# --------------------------------------------------------------------------
# process_hmi.py settings
# --------------------------------------------------------------------------
HMI_CON_EXTENSION = '.continuum.fits'
HMI_MAG_EXTENSION = '.Br.fits'
HMI_BMP_EXTENSION = '.bitmap.fits'
 
# Directory to read continuum/Br/bitmap files from — the same directory
# download_hmi.py downloads into.
HMI_PROCESS_DIR = HMI_DIR
 
# joblib.Parallel settings — https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
N_JOBS = 4
JOBLIB_VERBOSITY = 0
OUTER_BATCH_SIZE = 50
JOBLIB_BATCH_SIZE = 1

