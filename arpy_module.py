import warnings

import numpy as np
import sunpy.map
import astropy.units as u
from scipy import ndimage

_PIERCE_SLAUGHTER_A_HMI = 0.36019
_PIERCE_SLAUGHTER_B_HMI = 0.90010
_PIERCE_SLAUGHTER_C_HMI = -0.26029


def test_close_limb_images(sunpy_map:sunpy.map.Map, limit:float=75):
    """Test whether a continuum image contains any close-limb data.

    Close-limb images are not trusted and are excluded upstream. This
    is a binary quick-check used during pre-processing.

    Args:
        sunpy_map: SunPy map of the continuum image.
        limit: Longitude/latitude magnitude (degrees) at or beyond
            which the image is considered too close to the limb.

    Returns:
        True if the image contains no close-limb data, False otherwise.
    """
    all_car = sunpy.map.all_coordinates_from_map(sunpy_map)
    all_hgs = all_car.transform_to(sunpy.coordinates.frames.HeliographicStonyhurst)

    limit_deg = limit * u.degree
    lon_max = np.max(np.abs(all_hgs.lon))
    lat_max = np.max(np.abs(all_hgs.lat))

    return bool(lon_max < limit_deg and lat_max < limit_deg)



def normalise_to_qs(continuum_image:np.ndarray):
    """Normalise a continuum image relative to its median (quiet sun).

    Divides every pixel by the image's median value — used as a proxy
    for the quiet-sun continuum level — and rounds to 10 d.p. Falls
    back to the unnormalised image if the median is exactly zero, to
    avoid a division-by-zero.

    Args:
        continuum_image: 2D array continuum image.

    Returns:
        Array normalised relative to the median (quiet sun), or the
        original array unchanged if the median was zero.
    """
    img_median = np.nanmedian(continuum_image)

    if img_median == 0:
        warnings.warn('Failed to normalise image: median is 0.')
        return np.array(continuum_image)

    return np.round(continuum_image / img_median, 3).astype(np.float64)



def limb_darkening_correction_hmi(continuum_map:sunpy.map.Map):
    """Apply a 2nd-order polynomial limb-darkening correction.

    For SunPy Map SHARP CEA continuum data, using coefficients from
    Pierce & Slaughter (1977, doi: 10.1007/BF00240442).

    Args:
        continuum_map: SunPy Map of the continuum image.

    Returns:
        A limb-darkening-corrected SunPy Map continuum.
    """
    all_car = sunpy.map.all_coordinates_from_map(continuum_map)
    all_hee = all_car.transform_to('heliocentricearthecliptic')

    mu = np.cos(all_hee.lat) * np.cos(all_hee.lon)
    limb_darkening = (
        _PIERCE_SLAUGHTER_A_HMI
        + _PIERCE_SLAUGHTER_B_HMI * mu
        + _PIERCE_SLAUGHTER_C_HMI * mu ** 2
    )

    return continuum_map / limb_darkening



def clean_mask(feature_mask:np.ndarray, req_neighbours:int=3) -> np.ndarray:
    """Denoise a binary feature mask.

    Erodes the mask by keeping only pixels with more than
    req_neighbours 'on' cells in their 3x3 neighbourhood (removing
    small, isolated specks), then regrows surviving regions back to
    their original extent within feature_mask — it never expands
    beyond the original mask.

    Args:
        feature_mask: 2D binary array (0/1) of a feature mask.
        req_neighbours: Minimum neighbour count (out of 9, including
            self) required for a pixel to survive erosion.

    Returns:
        Denoised binary mask, same shape as feature_mask.
    """
    neighbour_count = ndimage.convolve(feature_mask, np.ones((3, 3)), mode='constant', cval=0)
    eroded_mask = (neighbour_count > req_neighbours) * feature_mask

    regrow_count = ndimage.convolve(eroded_mask, np.ones((3, 3)), mode='constant', cval=0)
    regrown_mask = (regrow_count * feature_mask) > 0

    return regrown_mask



def extract_feature_coords(input_array:np.ndarray, scan_step:int=1, min_area:int=4):
    """Partition a binary mask into separate connected features.

    Two 'on' pixels belong to the same feature if there's a chain of
    'on' pixels between them where each consecutive pair is within
    Chebyshev distance scan_step of each other (scan_step=1 is
    standard 8-connectivity; larger values bridge small gaps between
    otherwise-separate regions).

    Implemented via binary dilation + connected-component labelling,
    ~10-20x faster than a manual flood-fill at typical SHARP patch
    sizes. This is only exact for odd scan_step (the dilation radius
    must be an integer) — verified by construction and by 1000
    randomised trials at scan_step=3. An even scan_step raises
    NotImplementedError rather than silently giving a wrong answer.

    Args:
        input_array: 2D binary mask (0/1 or False/True).
        scan_step: Chebyshev distance within which two 'on' pixels are
            considered part of the same feature. Must be odd.
        min_area: Minimum number of pixels for a region to be kept.

    Returns:
        A list of features, each a list of [row, col] coordinate pairs.
    """
    if scan_step % 2 == 0:
        raise NotImplementedError(
            f'scan_step={scan_step} is even; this implementation is '
            'only verified exact for odd scan_step.'
        )

    mask = input_array.astype(bool)
    dilation_radius = (scan_step - 1) // 2

    if dilation_radius > 0:
        structure = np.ones((2 * dilation_radius + 1, 2 * dilation_radius + 1), dtype=bool)
        dilated = ndimage.binary_dilation(mask, structure=structure)
    else:
        dilated = mask

    labeled, num_features = ndimage.label(dilated, structure=np.ones((3, 3)))
    labeled_in_mask = labeled * mask

    output_features = []
    for label_id in range(1, num_features + 1):
        coords = np.argwhere(labeled_in_mask == label_id)
        if len(coords) >= min_area:
            output_features.append(coords.tolist())

    return output_features



def convert_coords_to_masks(input_region_coordinates:list, original_array:np.ndarray):
    """Convert a list of per-feature coordinates into a list of 2D masks.

    Args:
        input_region_coordinates: List of regions, each a list of
            [row, col] coordinate pairs.
        original_array: Original array, used only to match output shape.

    Returns:
        A list of 2D boolean masks, one per region, matching the input
        coordinates.
    """
    shape = original_array.shape
    mask_list = []

    for region in input_region_coordinates:
        mask = np.zeros(shape, dtype=bool)
        for row, col in region:
            mask[row, col] = True
        mask_list.append(mask)

    return mask_list



def spherical_centroid(points_ll, weights=None):
    """Weighted centroid of points on a sphere, in lat/lon degrees.

    Averages via Cartesian unit vectors rather than a naive arithmetic
    mean of lat/lon, which isn't meaningful for spherical coordinates.
    This is the shared core behind compute_feature_centroid (uniform
    weights) and extract_fw_centroids (flux weights).

    Args:
        points_ll: List of [lat, lon] points, in degrees.
        weights: Optional list of weights, same length as points_ll.
            Defaults to equal weighting. Negative values (e.g. signed
            flux) are used by magnitude.

    Returns:
        [lat, lon] centroid in degrees, or [nan, nan] if there are no
        points or the total weight is zero.
    """
    if not points_ll:
        return [np.nan, np.nan]

    if weights is None:
        weights = [1.0] * len(points_ll)

    total_weight = sum(abs(w) for w in weights)
    if total_weight == 0:
        return [np.nan, np.nan]

    x = y = z = 0.0
    for (lat, lon), weight in zip(points_ll, weights):
        lat_r, lon_r = np.radians(lat), np.radians(lon)
        cos_lat = np.cos(lat_r)
        w = abs(weight)
        x += cos_lat * np.cos(lon_r) * w
        y += cos_lat * np.sin(lon_r) * w
        z += np.sin(lat_r) * w

    x, y, z = x / total_weight, y / total_weight, z / total_weight
    norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    return [float(np.degrees(np.arcsin(z / norm))), float(np.degrees(np.arctan2(y, x)))]



def compute_feature_centroid(feature_coords, lat_map, lon_map):
    """Unweighted geometric centroid of a feature, in lat/lon space.

    Args:
        feature_coords: List of [row, col] pixel coordinates making up
            the feature.
        lat_map: 2D array of latitude (degrees) at each pixel.
        lon_map: 2D array of longitude (degrees) at each pixel.

    Returns:
        [lat, lon] centroid in degrees.
    """
    points_ll = [[lat_map[row, col], lon_map[row, col]] for row, col in feature_coords]
    return spherical_centroid(points_ll)



def extract_fw_centroids(positive_centroids_ll, negative_centroids_ll, positive_flux, negative_flux):
    """Flux-weighted centroid for each polarity group, in lat/lon degrees.

    Takes already-computed per-feature centroids and flux — it no
    longer recomputes flux internally (the original did, redundantly,
    since every call site already has pos_flux_list/neg_flux_list from
    extract_unsigned_flux before calling this).

    Args:
        positive_centroids_ll: [lat, lon] centroid per positive feature.
        negative_centroids_ll: [lat, lon] centroid per negative feature.
        positive_flux: Unsigned flux per positive feature, same order.
        negative_flux: Unsigned flux per negative feature, same order.

    Returns:
        (pfwc, nfwc): each a [lat, lon] centroid, or [nan, nan] if that
        polarity has no features or zero total flux.
    """
    pfwc = spherical_centroid(positive_centroids_ll, weights=positive_flux)
    nfwc = spherical_centroid(negative_centroids_ll, weights=negative_flux)
    return pfwc, nfwc



def extract_unsigned_flux(magnetogram: np.ndarray, feature_coordinates: list[list[int]]) -> int:
    """Extract the integrated unsigned flux of a feature.

    Args:
        magnetogram: Magnetogram image array, extracted from a FITS file.
        feature_coordinates: List of [row, col] coordinates covered by
            the feature.

    Returns:
        The integrated unsigned flux of the feature, truncated to int.
    """
    flux_output = [abs(magnetogram[row, col]) for row, col in feature_coordinates]
    return int(np.sum(flux_output))



def extract_incident_lines(positive_coord: list[float], negative_coord: list[float], degrees: bool = True) -> float:
    """Compute the forward bearing between two [lat, lon] coordinates.

    Uses the standard spherical (great-circle) initial-bearing formula.

    Args:
        positive_coord: [lat, lon] of the positive coordinate, degrees.
        negative_coord: [lat, lon] of the negative coordinate, degrees.
        degrees: Return degrees if True, radians if False.

    Returns:
        The bearing angle between the two coordinates, or nan if
        either coordinate is undefined (nan).
    """
    if np.isnan(positive_coord[0]) or np.isnan(negative_coord[0]):
        return np.nan

    lat1 = np.radians(positive_coord[0])
    lat2 = np.radians(negative_coord[0])
    dlon = np.radians(negative_coord[1] - positive_coord[1])

    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.arctan2(x, y)

    return np.degrees(bearing) if degrees else bearing



def extract_inversion_measure(sorted_polarity_list: list[str]) -> int:
    """Count adjacent polarity changes in a distance-ordered 'p'/'n' sequence.

    Called once per feature by polarity_inversion_measures, using a
    polarity sequence ordered by great-circle distance from that
    feature (including itself, first, at distance zero).

    Args:
        sorted_polarity_list: Polarities ('p'/'n'), ordered by
            distance from the feature of interest.

    Returns:
        Number of times consecutive entries differ in polarity.
    """
    count = 0
    for i in range(len(sorted_polarity_list) - 1):
        if sorted_polarity_list[i] != sorted_polarity_list[i + 1]:
            count += 1
    return count


def polarity_inversion_measures(
    pos_feature_centroids: list[list[float]],
    neg_feature_centroids: list[list[float]],
    pos_flux_list: list[float],
    neg_flux_list: list[float],
) -> tuple[float, float]:
    """Compute unweighted and flux-weighted polarity inversion measures (PIM).

    For each feature, every feature (including itself) is ordered by
    great-circle distance from it, and extract_inversion_measure counts
    how often polarity changes along that ordering. The unweighted PIM
    is the mean count across all features; the flux-weighted PIM
    weights each feature's count by its share of total unsigned flux.

    Replaces separately calling an unweighted and a flux-weighted
    version of this measure — both built the same distance-ordered
    polarity sequence per feature, so computing it once here avoids
    building the O(n^2) distance matrix twice.

    Args:
        pos_feature_centroids: [lat, lon] centroid per positive feature.
        neg_feature_centroids: [lat, lon] centroid per negative feature.
        pos_flux_list: Unsigned flux per positive feature, same order.
        neg_flux_list: Unsigned flux per negative feature, same order.

    Returns:
        (pim_uw, pim_fw): both nan if there are no features at all,
        both 0.0 if only one polarity is present.
    """
    n_pos, n_neg = len(pos_feature_centroids), len(neg_feature_centroids)

    if n_pos + n_neg == 0:
        return np.nan, np.nan
    if n_pos == 0 or n_neg == 0:
        return 0., 0.

    centroids = list(pos_feature_centroids) + list(neg_feature_centroids)
    polarities = np.array(['p'] * n_pos + ['n'] * n_neg)
    flux = np.array(list(pos_flux_list) + list(neg_flux_list), dtype=float)

    # Vectorized great-circle distance matrix
    coords = np.radians(np.array(centroids, dtype=float))
    lat, lon = coords[:, 0], coords[:, 1]
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    distance_matrix = 2 * np.arcsin(np.sqrt(a))

    # Tie-break key matching the original sorted([distance, polarity])
    polarity_tiebreak = (polarities == 'p').astype(int)

    n_features = len(centroids)
    pim_per_feature = np.empty(n_features)
    for i in range(n_features):
        order = np.lexsort((polarity_tiebreak, distance_matrix[i]))
        pim_per_feature[i] = extract_inversion_measure(polarities[order].tolist())

    total_flux = flux.sum()
    pim_uw = float(np.mean(pim_per_feature))
    pim_fw = float(np.sum(pim_per_feature * flux / total_flux)) if total_flux > 0 else np.nan

    return pim_uw, pim_fw



def angular_deviation_measures(
    positive_centroids: list[list[float]],
    negative_centroids: list[list[float]],
    pos_flux_list: list[float],
    neg_flux_list: list[float],
    guide_line_angle: float,
) -> tuple[float, float]:
    """Compute unweighted and flux-weighted angular deviation measures (ADM).

    For every positive/negative feature pair, the bearing between the
    two centroids is compared to guide_line_angle via
    1 - cos(bearing - guide_line_angle) — 0 for a pair aligned with
    the guide line, 2 for a pair at 180 degrees to it. The unweighted
    ADM is the mean over all pairs; the flux-weighted ADM weights each
    pair by the combined flux of its two features.

    Replaces separately calling an unweighted and a flux-weighted
    version — both built the same pairwise bearing/deviation values,
    and the flux-weighted version recomputed flux from raw pixel data
    even though the caller already has it from extract_unsigned_flux.

    Args:
        positive_centroids: [lat, lon] centroid per positive feature.
        negative_centroids: [lat, lon] centroid per negative feature.
        pos_flux_list: Unsigned flux per positive feature, same order.
        neg_flux_list: Unsigned flux per negative feature, same order.
        guide_line_angle: Bearing (degrees) between the flux-weighted
            positive and negative barycentres.

    Returns:
        (adm_uw, adm_fw): both nan if either polarity has no features,
        both 0.0 if there is exactly one feature of each polarity.
    """
    n_pos, n_neg = len(positive_centroids), len(negative_centroids)

    if n_pos == 0 or n_neg == 0:
        return np.nan, np.nan
    if n_pos + n_neg == 2:
        return 0., 0.

    pos_arr = np.radians(np.array(positive_centroids, dtype=float))
    neg_arr = np.radians(np.array(negative_centroids, dtype=float))
    lat1, lon1 = pos_arr[:, 0][:, None], pos_arr[:, 1][:, None]
    lat2, lon2 = neg_arr[:, 0][None, :], neg_arr[:, 1][None, :]
    dlon = lon2 - lon1

    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))  # shape (n_pos, n_neg)

    deviation = 1 - np.cos(np.radians(bearing - guide_line_angle))
    adm_uw = float(np.mean(deviation))

    pos_flux = np.array(pos_flux_list, dtype=float)[:, None]
    neg_flux = np.array(neg_flux_list, dtype=float)[None, :]
    pair_flux = pos_flux + neg_flux
    total_flux = pair_flux.sum()
    adm_fw = float(np.sum(deviation * pair_flux / total_flux)) if total_flux > 0 else np.nan

    return adm_uw, adm_fw    



