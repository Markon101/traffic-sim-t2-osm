from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def compute_transform(
    bounds: Tuple[float, float, float, float],
    screen_size: Tuple[int, int],
    padding: int = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return affine transform (scale, translation) from world to screen coordinates."""

    min_x, min_y, max_x, max_y = bounds
    width = max_x - min_x
    height = max_y - min_y

    screen_w, screen_h = screen_size
    usable_w = screen_w - 2 * padding
    usable_h = screen_h - 2 * padding

    scale = min(usable_w / width, usable_h / height)
    tx = padding - min_x * scale + (usable_w - width * scale) * 0.5
    ty = padding - min_y * scale + (usable_h - height * scale) * 0.5
    return scale, np.array([tx, ty])


def world_to_screen(
    coord: Tuple[float, float],
    scale: float,
    translation: np.ndarray,
    flip_y: bool = True,
    screen_height: int | None = None,
) -> Tuple[int, int]:
    x, y = coord
    point = np.array([x, y]) * scale + translation
    if flip_y and screen_height is not None:
        point[1] = screen_height - point[1]
    return int(point[0]), int(point[1])


def line_to_points(
    line: Iterable[Tuple[float, float]],
    scale: float,
    translation: np.ndarray,
    flip_y: bool = True,
    screen_height: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(list(line))
    transformed = coords * scale + translation
    if flip_y and screen_height is not None:
        transformed[:, 1] = screen_height - transformed[:, 1]
    return transformed[:, 0], transformed[:, 1]
