"""
Grid utilities: map lat/lon coordinates to integer grid cell indices.

Grid definition (left-closed, right-open intervals):
    lat_idx = floor(lat / step)
    lon_idx = floor(lon / step)

    lat range: [lat_idx * step, (lat_idx + 1) * step)
    lon range: [lon_idx * step, (lon_idx + 1) * step)

This ensures every point falls in exactly one cell, including negative
coordinates (Python's math.floor handles negative values correctly, unlike
simple int() truncation in C/C++).
"""

import math
from typing import Tuple


def compute_grid_id(lat: float, lon: float, step: float = 0.001) -> Tuple[int, int]:
    """
    Compute the integer grid cell index for a (lat, lon) coordinate.

    Args:
        lat:  latitude  in decimal degrees.
        lon:  longitude in decimal degrees.
        step: cell size in degrees (default 0.001 ≈ 111 m at the equator).

    Returns:
        (lat_idx, lon_idx) – a pair of integers uniquely identifying the cell.
    """
    lat_idx = math.floor(lat / step)
    lon_idx = math.floor(lon / step)
    return (lat_idx, lon_idx)


def grid_id_to_str(grid_id: Tuple[int, int]) -> str:
    """
    Encode a grid ID tuple as a string key suitable for dict / sketch lookup.

    Format: "<lat_idx>_<lon_idx>"
    Works correctly for negative indices (e.g., "-30_-97").
    """
    return f"{grid_id[0]}_{grid_id[1]}"


def grid_str_to_id(key: str) -> Tuple[int, int]:
    """
    Decode a grid key string back to (lat_idx, lon_idx).

    Splits on the *first* underscore that is not part of a negative sign,
    i.e., the '_' that separates lat_idx from lon_idx.
    For example: "-30_-97" → (-30, -97), "30219_-97759" → (30219, -97759).
    """
    # Find the separator: the '_' that follows the lat part.
    # lat_idx may be negative, so skip a leading '-' before searching.
    start = 1 if key.startswith("-") else 0
    sep = key.index("_", start)
    lat_idx = int(key[:sep])
    lon_idx = int(key[sep + 1:])
    return (lat_idx, lon_idx)


def grid_id_to_range(grid_id: Tuple[int, int], step: float = 0.001) -> dict:
    """
    Convert a grid ID to its geographic bounding box.

    Returns a dict with keys: lat_min, lat_max, lon_min, lon_max.
    All intervals are left-closed, right-open: [min, max).
    """
    lat_idx, lon_idx = grid_id
    return {
        "lat_min": lat_idx * step,
        "lat_max": (lat_idx + 1) * step,
        "lon_min": lon_idx * step,
        "lon_max": (lon_idx + 1) * step,
    }
