"""Named quarter-turn transforms for a single cubie orientation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Matrix = NDArray[np.integer]


def x_plus_90() -> Matrix:
    """Positive 90-degree rotation around world x, by the right-hand rule."""

    return np.array(
        [
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ],
        dtype=int,
    )


def y_plus_90() -> Matrix:
    """Positive 90-degree rotation around world y, by the right-hand rule."""

    return np.array(
        [
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0],
        ],
        dtype=int,
    )


def z_plus_90() -> Matrix:
    """Positive 90-degree rotation around world z, by the right-hand rule."""

    return np.array(
        [
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ],
        dtype=int,
    )


def x_minus_90() -> Matrix:
    """Negative 90-degree rotation around world x."""

    return x_plus_90().T


def y_minus_90() -> Matrix:
    """Negative 90-degree rotation around world y."""

    return y_plus_90().T


def z_minus_90() -> Matrix:
    """Negative 90-degree rotation around world z."""

    return z_plus_90().T


TRANSFORMS: dict[str, Matrix] = {
    "x+": x_plus_90(),
    "x-": x_minus_90(),
    "y+": y_plus_90(),
    "y-": y_minus_90(),
    "z+": z_plus_90(),
    "z-": z_minus_90(),
}
