"""
Template for machine-specific settings.

Copy this file to local_settings.py and fill in your own values.
local_settings.py is gitignored — it should never be committed, since
it holds paths and credentials specific to each machine.
"""

# Directory for downloaded HMI FITS files. Must exist and be writable.
HMI_DIR = '/path/to/your/hmi/directory/'

# Email registered with JSOC for data export requests.
# Register at: http://jsoc.stanford.edu/ajax/register_email.html
JSOC_REGISTERED_EMAIL = 'you@example.com'

# Directory for downloaded GONG FITS files. Must exist and be writable.
GONG_DIR = '/path/to/your/gong/directory/'

# Directory for processed JSON output files.
OUTPUT_DIR = '/path/to/your/output/directory/'


