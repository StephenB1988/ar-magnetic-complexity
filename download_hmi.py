"""Download near-real-time SDO/HMI SHARP data from JSOC.

A single JSOC search covers every HARP present in the lookback window,
rather than one search per HARP. For each HARP found, only the
N_RECENT_TIMESTEPS most recent records are kept, then everything is
downloaded in one Fido.fetch call.
"""

import datetime
import os
import time

from sunpy.net import Fido, attrs as a

import config


def get_time_range(n_hours):
    """Return (t_start, t_end) as timezone-aware UTC datetimes."""
    t_end = datetime.datetime.now(datetime.timezone.utc)
    t_start = t_end - datetime.timedelta(hours=n_hours)
    return t_start, t_end


def search_all_harps(t_start, t_end):
    """Single JSOC search covering every HARP present in the window."""
    return Fido.search(
        a.Time(t_start, t_end),
        a.jsoc.Series('hmi.sharp_cea_720s_nrt'),
        a.jsoc.Segment('continuum'),
        a.jsoc.Segment('Br'),
        a.jsoc.Segment('bitmap'),
        a.jsoc.Notify(config.JSOC_REGISTERED_EMAIL),
    )


def count_existing_files(scan_dir, harp_num):
    """Count files in scan_dir belonging to the given HARP number."""
    count = 0
    for entry in os.scandir(scan_dir):
        if entry.is_file() and f'.{harp_num}.' in entry.name:
            count += 1
    return count


def fetch_with_retries(rows):
    """Download the given rows, retrying on failure. Returns success flag."""
    for attempt in range(config.MAX_RETRIES):
        try:
            print(f'Download attempt {attempt + 1}/{config.MAX_RETRIES}...')
            Fido.fetch(rows, path=config.DOWNLOAD_DIR, max_conn=3)
            return True
        except Exception as exc:
            print(f'Download failed: {type(exc).__name__}: {exc}')
            if attempt < config.MAX_RETRIES - 1:
                print(f'Retrying in {config.RETRY_WAIT_SECONDS} seconds...')
                time.sleep(config.RETRY_WAIT_SECONDS)
    return False


def main():
    t_start, t_end = get_time_range(config.N_HOURS)

    res = search_all_harps(t_start, t_end)

    harp_nums = sorted({int(row['HARPNUM']) for row in res[0]})
    print('HARPs found:', harp_nums)
    print(f'Fetching records found in the last {config.N_HOURS} hour(s).')

    with open(config.DOWNLOAD_LOG_FILE, 'a') as log:
        log.write(
            '\n----- New query launched on '
            + datetime.datetime.now(datetime.timezone.utc).strftime('%d/%m/%Y %H:%M UTC')
            + ' -----\n'
        )
        log.write(f'HARPs found: {harp_nums}\n')

    download_success = fetch_with_retries(res)

    if not download_success:
        with open(config.DOWNLOAD_LOG_FILE, 'a') as log:
            log.write(f'Download FAILED after {config.MAX_RETRIES} attempts\n')
        time.sleep(config.FAILURE_WAIT_SECONDS)
        raise RuntimeError(f'Download failed after {config.MAX_RETRIES} attempts.')

    with open(config.DOWNLOAD_LOG_FILE, 'a') as log:
        for harp_num in harp_nums:
            total = count_existing_files(config.SCAN_DIR, harp_num)
            log.write(f'nrt HARP: {harp_num}. Files on disk: {total}\n')

    print('Complete. See log for more details:', config.DOWNLOAD_LOG_FILE)


if __name__ == '__main__':
    main()
