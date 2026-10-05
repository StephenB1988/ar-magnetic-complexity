"""
Process PIM/ADM values for downloaded SDO/HMI SHARP triplets.

For each HARP found in config.HMI_PROCESS_DIR, matches continuum, Br, and
bitmap files into triplets by (HARPNUM, T_REC), extracts magnetic
features, computes polarity-inversion and angular-deviation measures,
and writes the results to one JSON file per HARP in config.OUTPUT_DIR.

NRT vs definitive data is detected automatically from each filename's
series name ('hmi.sharp_cea_720s_nrt' vs 'hmi.sharp_cea_720s').
"""

import datetime
import json
import os
from pathlib import Path

import numpy as np
import sunpy.map
import sunpy.coordinates.frames
from astropy.io import fits
from astropy.time import Time
from joblib import Parallel, delayed

from arpy import (
    test_close_limb_images,
    normalise_to_qs,
    limb_darkening_correction_hmi,
    clean_mask,
    extract_feature_coords,
    convert_coords_to_masks,
    compute_feature_centroid,
    spherical_centroid,
    extract_unsigned_flux,
    extract_fw_centroids,
    extract_incident_lines,
    polarity_inversion_measures,
    angular_deviation_measures
)

import config


def ensure_leap_second_data():
    """Force astropy to fetch leap-second data once, up front.

    Avoids every parallel worker separately triggering the same
    download on first use of Time.
    """
    Time.now()


def parse_sharp_filename(filename):
    """Parse a JSOC SHARP export filename into its component fields.

    Expected format: '<inst>.<series>.<harpnum>.<trec>.<segment>.fits',
    e.g. 'hmi.sharp_cea_720s_nrt.7371.20231001_000000_TAI.Br.fits'.
    Assumes JSOC's dot-delimited export naming and that T_REC values in
    filenames don't themselves contain dots. Not yet verified against
    an actual downloaded file.
    """
    parts = filename.split('.')
    return {
        'series': f'{parts[0]}.{parts[1]}',
        'harpnum': int(parts[2]),
        'trec': parts[3],
        'segment': parts[4],
        'is_nrt': '_nrt' in parts[1],
    }


def find_triplets(data_dir):
    """Group continuum/Br/bitmap files in data_dir by (harpnum, trec).

    Returns (triplets, is_nrt): triplets maps harpnum to a time-sorted
    list of (continuum_file, mag_file, bmp_file) tuples; is_nrt is read
    from whichever file was parsed (assumes the directory isn't a mix
    of NRT and definitive data).
    """
    by_key = {}
    is_nrt = None
    extensions = (config.HMI_CON_EXTENSION, config.HMI_MAG_EXTENSION,
                  config.HMI_BMP_EXTENSION)

    for entry in os.scandir(data_dir):
        if not entry.is_file():
            continue

        for extension in extensions:
            if entry.name.endswith(extension):
                parsed = parse_sharp_filename(entry.name)
                if is_nrt is None:
                    is_nrt = parsed['is_nrt']
                key = (parsed['harpnum'], parsed['trec'])
                by_key.setdefault(key, {})[extension] = entry.name
                break

    triplets = {}
    for (harp_num, trec), files in by_key.items():
        if all(ext in files for ext in extensions):
            triplet = tuple(files[ext] for ext in extensions)
            triplets.setdefault(harp_num, []).append((trec, triplet))

    for harp_num, entries in triplets.items():
        entries.sort(key=lambda item: item[0])
        triplets[harp_num] = [triplet for _, triplet in entries]

    return triplets, bool(is_nrt)


def stem(filename, extension):
    return filename[:-(len(extension))]


def filter_by_bitmap(co_list, im_list, centroid_list, bmp_mask):
    filtered_co, filtered_im, filtered_centroids = [], [], []
    for co, im, centroid in zip(co_list, im_list, centroid_list):
        if np.any(im * bmp_mask):
            filtered_co.append(co)
            filtered_im.append(im)
            filtered_centroids.append(centroid)
    return filtered_co, filtered_im, filtered_centroids


def process_single_triplet(image_triplet, data_dir):
    """Feature extraction and measures calculation for a single triplet.

    Returns {'success': bool, 'log_entry': str, 'measures': {...}} on
    success, or without 'measures' on failure.
    """
    con_map = None

    try:
        con_map = sunpy.map.Map(os.path.join(data_dir, image_triplet[0]))
    except Exception as error:
        return {
            'success': False,
            'log_entry': f'{stem(image_triplet[0], config.HMI_CON_EXTENSION)}: '
                         f'FAILED, exception: {type(error).__name__}',
        }

    try:
        if not test_close_limb_images(con_map, limit=config.LIMB_LIMIT):
            return {
                'success': False,
                'log_entry': f'{stem(image_triplet[0], config.HMI_CON_EXTENSION)}: '
                             f'IGNORED (close-limb)',
                'close_limb': True,
            }
    except Exception as error:
        del con_map
        return {
            'success': False,
            'log_entry': f'{stem(image_triplet[0], config.HMI_CON_EXTENSION)}: '
                         f'Close-limb test FAILED, exception: {type(error).__name__}',
        }

    try:
        mag = fits.getdata(os.path.join(data_dir, image_triplet[1]))
        bmp = fits.getdata(os.path.join(data_dir, image_triplet[2]))

        # Pull everything needed from con_map out now, so it can be
        # freed before the heavier feature-extraction work below.
        met = con_map.meta
        all_car = sunpy.map.all_coordinates_from_map(con_map)
        con = normalise_to_qs(limb_darkening_correction_hmi(con_map).data)
    except Exception as error:
        return {
            'success': False,
            'log_entry': f'{stem(image_triplet[0], config.HMI_CON_EXTENSION)}: '
                         f'FAILED, exception: {type(error).__name__}',
        }
    finally:
        con_map = None  # drop the reference, but keep the name bound

    try:
        all_hs = all_car.transform_to(sunpy.coordinates.frames.HeliographicStonyhurst)
        lat_map = all_hs.lat.value
        lon_map = all_hs.lon.value
        del all_car, all_hs

        threshold_mask = con <= config.FEATURE_INTENSITY

        polarity_map = (np.zeros_like(con)
                         + ((mag > config.GAUSS_THRESHOLD) * 1)
                         + ((mag < -config.GAUSS_THRESHOLD) * -1))
        positive_mask_pre = threshold_mask * (polarity_map > 0)
        negative_mask_pre = threshold_mask * (polarity_map < 0)
        del threshold_mask, polarity_map, con

        positive_mask = clean_mask(positive_mask_pre)
        negative_mask = clean_mask(negative_mask_pre)
        del positive_mask_pre, negative_mask_pre

        feature_masks_co_p = extract_feature_coords(
            positive_mask, config.PIXEL_SCAN_STEP, config.MIN_AREA)
        feature_masks_co_n = extract_feature_coords(
            negative_mask, config.PIXEL_SCAN_STEP, config.MIN_AREA)

        feature_masks_im_p = convert_coords_to_masks(feature_masks_co_p, positive_mask)
        feature_masks_im_n = convert_coords_to_masks(feature_masks_co_n, negative_mask)
        del positive_mask, negative_mask

        feature_centroid_co_p = [
            compute_feature_centroid(f, lat_map, lon_map) for f in feature_masks_co_p]
        feature_centroid_co_n = [
            compute_feature_centroid(f, lat_map, lon_map) for f in feature_masks_co_n]

        bmp_mask = bmp >= 30
        feature_masks_co_p, feature_masks_im_p, feature_centroid_co_p = filter_by_bitmap(
            feature_masks_co_p, feature_masks_im_p, feature_centroid_co_p, bmp_mask)
        feature_masks_co_n, feature_masks_im_n, feature_centroid_co_n = filter_by_bitmap(
            feature_masks_co_n, feature_masks_im_n, feature_centroid_co_n, bmp_mask)
        del feature_masks_im_p, feature_masks_im_n, bmp_mask, bmp

        p_centroids = list(feature_centroid_co_p)
        n_centroids = list(feature_centroid_co_n)

        pos_flux_list = [extract_unsigned_flux(mag, mask) for mask in feature_masks_co_p]
        neg_flux_list = [extract_unsigned_flux(mag, mask) for mask in feature_masks_co_n]

        pfwc, nfwc = extract_fw_centroids(p_centroids, n_centroids, pos_flux_list, neg_flux_list)
        guide_line_angle = extract_incident_lines(pfwc, nfwc, degrees=True)

        # Global AR centroid: flux-weighted spherical mean of every
        # feature's lat/lon centroid (both polarities combined).
        all_centroids_ll = p_centroids + n_centroids
        all_flux = pos_flux_list + neg_flux_list
        global_centroid = spherical_centroid(all_centroids_ll, weights=all_flux)

        pim_uw, pim_fw = polarity_inversion_measures(p_centroids, n_centroids, pos_flux_list, neg_flux_list)
        adm_uw, adm_fw = angular_deviation_measures(p_centroids, n_centroids, pos_flux_list, neg_flux_list, guide_line_angle)

        del mag

        return {
            'success': True,
            'log_entry': f'{stem(image_triplet[0], config.HMI_CON_EXTENSION)}: SUCCESS.',
            'measures': {
                'time': datetime.datetime.strptime(
                    met['t_rec'][0:19], '%Y.%m.%d_%H:%M:%S').isoformat(),
                'global_centroid': global_centroid,
                'pim_uw': float(pim_uw),
                'pim_fw': float(pim_fw),
                'adm_uw': float(adm_uw),
                'adm_fw': float(adm_fw),
            },
        }

    except Exception as error:
        if con_map is not None:
            del con_map
        return {
            'success': False,
            'log_entry': f'{stem(image_triplet[0], config.HMI_CON_EXTENSION)}: '
                         f'FAILED, exception: {type(error).__name__}',
        }


def load_existing_measures(measures_path):
    """Load previously written measures and the set of trecs covered.

    'trec' is stored per entry purely so reruns can tell which
    time-steps are already done — it isn't one of the six fields you
    listed as important, so drop it before using the data downstream
    if you'd rather it not be there; it's cheap to strip at load time.
    """
    if not os.path.exists(measures_path):
        return [], set()

    with open(measures_path, 'r') as f:
        existing_data = json.load(f)

    existing_measures = existing_data.get('measures', [])
    processed_trecs = {m['trec'] for m in existing_measures if 'trec' in m}
    return existing_measures, processed_trecs


def process_harp(harp_num, triplet_list, data_dir, output_dir, is_nrt):
    """Process only not-yet-processed triplets for one HARP and update its JSON."""
    measures_path = os.path.join(output_dir, f'{harp_num}_measures.json')

    existing_measures, processed_trecs = load_existing_measures(measures_path)

    new_triplets = [
        triplet for triplet in triplet_list
        if parse_sharp_filename(triplet[0])['trec'] not in processed_trecs
    ]

    if not new_triplets:
        print(f'HARP {harp_num}: no new triplets, skipping.')
        return

    print(f'Processing HARP {harp_num}: {len(new_triplets)} new triplet(s), '
          f'{len(triplet_list) - len(new_triplets)} already done.')
    time_start = datetime.datetime.now()

    log_entries = []
    err_entries = []
    results_out = []

    for batch_start in range(0, len(new_triplets), config.OUTER_BATCH_SIZE):
        batch_end = min(batch_start + config.OUTER_BATCH_SIZE, len(new_triplets))
        batch = new_triplets[batch_start:batch_end]

        results = Parallel(
            n_jobs=config.N_JOBS, verbose=config.JOBLIB_VERBOSITY,
            batch_size=config.JOBLIB_BATCH_SIZE, prefer='threads',
        )(
            delayed(process_single_triplet)(triplet, data_dir)
            for triplet in batch
        )

        for triplet, result in zip(batch, results):
            log_entries.append(result['log_entry'])
            if result['success']:
                measure = result['measures']
                measure['trec'] = parse_sharp_filename(triplet[0])['trec']
                results_out.append(measure)
            elif 'close_limb' not in result:
                err_entries.append(result['log_entry'])

    harp_log_file = os.path.join(output_dir, f'{harp_num}_log.log')
    with open(harp_log_file, 'a') as f:
        f.write('\n'.join(log_entries) + '\n')

    if err_entries:
        master_log_file = os.path.join(output_dir, 'err.log')
        with open(master_log_file, 'a') as f:
            f.write('\n'.join(err_entries) + '\n')

    all_measures = existing_measures + results_out
    all_measures.sort(key=lambda m: m['time'])

    output_data = {
        'harpnum': harp_num,
        'is_nrt': is_nrt,
        'measures': all_measures,
    }
    with open(measures_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    time_taken = datetime.datetime.now() - time_start
    print(f'HARP {harp_num} complete. Time taken: {str(time_taken).split(".")[0]} '
          f'({len(results_out)} new, {len(all_measures)} total).')


def main():
    ensure_leap_second_data()

    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    triplets, is_nrt = find_triplets(config.HMI_PROCESS_DIR)
    print(f'Data detected as: {"NRT" if is_nrt else "definitive"}')
    print('HARPs found:', sorted(triplets.keys()))

    for harp_num, triplet_list in triplets.items():
        if triplet_list:
            process_harp(harp_num, triplet_list, config.HMI_PROCESS_DIR,
                         config.OUTPUT_DIR, is_nrt)


if __name__ == '__main__':
    main()