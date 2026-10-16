"""Download co-temporal GONG magnetogram/continuum pairs over HTTP.

For each GONG site, the most recent continuum timestamps are tried in
turn (newest first, up to config.GONG_MAX_CANDIDATES attempts). For
each one, a magnetogram at the exact same timestamp is looked for; if
found, both files are downloaded and checked against the FITS header's
VALID keyword before being accepted as a genuine co-temporal pair.

Both products come from the same NSO HTTP archive and the same
underlying one-minute acquisition cycle at each site, so an exact
timestamp match is a meaningful signal of true co-temporality.
"""

import datetime
import gzip
import os
import shutil
import urllib.error
import urllib.request

from astropy.io import fits
from bs4 import BeautifulSoup

import config


def clean_existing_fits(path):
    """Delete pre-existing .fits/.fits.gz files from the download directory."""
    for entry in os.scandir(path):
        if entry.is_file() and ('.fits' in entry.name):
            os.remove(entry.path)


def parse_gong_filename(filename):
    """Parse a GONG quicklook filename into its component fields.

    Expected format: '<site><product>YYMMDDtHHMM.fits[.gz]', e.g.
    'bbbqa261015t1514.fits.gz' -> site 'bb', product 'bqa', timestamp
    2026-10-15 15:14.
    """
    name = filename
    for ext in ('.fits.gz', '.fits'):
        if name.endswith(ext):
            name = name[:-len(ext)]
            break

    site_code = name[:2]
    product_code = name[2:5]
    timestamp = datetime.datetime.strptime(name[5:], '%y%m%dt%H%M')

    return {'site_code': site_code, 'product_code': product_code, 'timestamp': timestamp}


def build_day_url(site_code, product_code, date):
    """Build the GONG archive URL for a given site/product/date."""
    year_month = date.strftime('%Y%m')
    yymmdd = date.strftime('%y%m%d')
    return f'{config.GONG_BASE_URL}/{product_code}/{year_month}/{site_code}{product_code}{yymmdd}/'


def fetch_directory_listing(url):
    """Return filenames of .fits.gz links at a GONG directory URL.

    Returns an empty list if the directory doesn't exist (e.g. HTTP 404)
    or contains no matching files.
    """
    try:
        response = urllib.request.urlopen(url)
    except urllib.error.URLError:
        return []

    html = response.read().decode('utf-8')
    soup = BeautifulSoup(html, 'html.parser')

    return [
        link.get('href') for link in soup.find_all('a')
        if link.get('href', '').endswith('.fits.gz')
    ]


def get_candidates(site_code, product_code, reference_date, lookback_days):
    """Get (timestamp, filename, directory_url) for a site/product.

    Covers reference_date and lookback_days-1 prior days, sorted by
    timestamp descending (most recent first).
    """
    candidates = []

    for days_ago in range(lookback_days):
        date = reference_date - datetime.timedelta(days=days_ago)
        url = build_day_url(site_code, product_code, date)
        for filename in fetch_directory_listing(url):
            timestamp = parse_gong_filename(filename)['timestamp']
            candidates.append((timestamp, filename, url))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def download_file(directory_url, filename, dest_dir):
    """Download a single file, returning its local path."""
    dest_path = os.path.join(dest_dir, filename)
    urllib.request.urlretrieve(directory_url + filename, dest_path)
    return dest_path


def is_fits_valid(local_path):
    """Check a downloaded GONG FITS file's VALID header keyword.

    astropy.io.fits reads gzip-compressed FITS files transparently, so
    this works on the still-compressed magnetogram as well as the
    decompressed continuum. A missing keyword is treated as invalid,
    to fail safe rather than silently accept unrecognised data.
    """
    with fits.open(local_path) as hdul:
        return bool(hdul[0].header.get('VALID', False))


def decompress_gz(gz_path):
    """Decompress a .gz file in place, removing the original, returning the new path."""
    fits_path = gz_path[:-len('.gz')]
    with gzip.open(gz_path, 'rb') as f_in, open(fits_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(gz_path)
    return fits_path


def find_matching_pair(site_code, reference_date, dest_dir):
    """Find and download the most recent valid, co-temporal pair for a site.

    Tries the GONG_MAX_CANDIDATES most recent continuum timestamps,
    newest first. For each, looks for a magnetogram at the exact same
    timestamp, downloads both if found, and checks VALID on each.
    Rejected candidates (no timestamp match, or failed VALID) are
    cleaned up immediately and don't count differently against the
    GONG_MAX_CANDIDATES budget than a straightforward timestamp miss.

    Returns (timestamp, con_path, mag_path) on success, or None if no
    valid pair was found within the attempt budget.
    """
    con_candidates = get_candidates(
        site_code, config.GONG_CON_PRODUCT, reference_date, config.GONG_LOOKBACK_DAYS
    )[:config.GONG_MAX_CANDIDATES]

    mag_candidates = get_candidates(
        site_code, config.GONG_MAG_PRODUCT, reference_date, config.GONG_LOOKBACK_DAYS
    )
    mag_by_timestamp = {ts: (filename, url) for ts, filename, url in mag_candidates}

    for con_timestamp, con_filename, con_url in con_candidates:
        if con_timestamp not in mag_by_timestamp:
            continue

        mag_filename, mag_url = mag_by_timestamp[con_timestamp]

        con_gz_path = download_file(con_url, con_filename, dest_dir)
        mag_gz_path = download_file(mag_url, mag_filename, dest_dir)

        con_valid = is_fits_valid(con_gz_path)
        mag_valid = is_fits_valid(mag_gz_path)

        if con_valid and mag_valid:
            con_path = decompress_gz(con_gz_path)
            return con_timestamp, con_path, mag_gz_path  # magnetogram stays .fits.gz

        os.remove(con_gz_path)
        os.remove(mag_gz_path)

    return None


def main():
    if config.GONG_CLEAN_BEFORE_DOWNLOAD:
        clean_existing_fits(config.GONG_DIR)

    reference_date = datetime.datetime.now()
    downloaded, skipped = [], []

    with open(config.GONG_DOWNLOAD_LOG_FILE, 'a') as log:
        log.write(
            '\n----- New query launched on '
            + reference_date.strftime('%d/%m/%Y %H:%M')
            + ' -----\n'
        )

    for site_name, site_code in config.GONG_SITES.items():
        print(f'Processing site: {site_name}                  ', end='\r')
        result = find_matching_pair(site_code, reference_date, config.GONG_DIR)

        if result is None:
            skipped.append(site_name)
            continue

        timestamp, con_path, mag_path = result
        downloaded.append((site_name, timestamp, con_path, mag_path))

    print('GONG download complete.                                ')
    print()
    print(f'Downloaded {len(downloaded)} co-temporal pair(s):')
    for site_name, timestamp, con_path, mag_path in downloaded:
        print(f'  {site_name:12s} {timestamp.strftime("%Y-%m-%d %H:%M")}  '
              f'{os.path.basename(con_path)}  {os.path.basename(mag_path)}')

    if skipped:
        print(f'Skipped {len(skipped)} site(s) (no valid co-temporal pair found): {skipped}')

    with open(config.GONG_DOWNLOAD_LOG_FILE, 'a') as log:
        for site_name, timestamp, con_path, mag_path in downloaded:
            log.write(f'{site_name}: {timestamp.isoformat()}: SUCCESS.\n')
        for site_name in skipped:
            log.write(f'{site_name}: no valid co-temporal pair found.\n')


if __name__ == '__main__':
    main()
