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


# ── 3-cubie block matrices (9×9) ──────────────────────────────────────────────
# State for 3 cubies is a (9×3) matrix: cubie k occupies rows 3k..3k+2.
# All ops are left-multiplication: new_state = T @ state.

_I = np.eye(3, dtype=int)
_Z = np.zeros((3, 3), dtype=int)


# Permutations ─────────────────────────────────────────────────────────────────

def swap_0_1() -> Matrix:
    """Swap cubies 0 and 1."""
    return np.block([[_Z, _I, _Z], [_I, _Z, _Z], [_Z, _Z, _I]])


def swap_1_2() -> Matrix:
    """Swap cubies 1 and 2."""
    return np.block([[_I, _Z, _Z], [_Z, _Z, _I], [_Z, _I, _Z]])


def swap_0_2() -> Matrix:
    """Swap cubies 0 and 2."""
    return np.block([[_Z, _Z, _I], [_Z, _I, _Z], [_I, _Z, _Z]])


def cycle_fwd() -> Matrix:
    """Cycle forward: 0 → 1 → 2 → 0."""
    return np.block([[_Z, _Z, _I], [_I, _Z, _Z], [_Z, _I, _Z]])


def cycle_bwd() -> Matrix:
    """Cycle backward: 0 → 2 → 1 → 0."""
    return np.block([[_Z, _I, _Z], [_Z, _Z, _I], [_I, _Z, _Z]])


# Orientation-on-slot ──────────────────────────────────────────────────────────

def orient_at_0(t: Matrix) -> Matrix:
    """Apply rotation t only to cubie 0."""
    return np.block([[t, _Z, _Z], [_Z, _I, _Z], [_Z, _Z, _I]])


def orient_at_1(t: Matrix) -> Matrix:
    """Apply rotation t only to cubie 1."""
    return np.block([[_I, _Z, _Z], [_Z, t, _Z], [_Z, _Z, _I]])


def orient_at_2(t: Matrix) -> Matrix:
    """Apply rotation t only to cubie 2."""
    return np.block([[_I, _Z, _Z], [_Z, _I, _Z], [_Z, _Z, t]])


# ── 9-cubie face block matrices (27×27) ───────────────────────────────────────
# State for 9 cubies is a (27×3) matrix: cubie k occupies rows 3k..3k+2.
# Face layout (reading order):
#   0 | 1 | 2
#   3 | 4 | 5
#   6 | 7 | 8
# CW rotation (viewed from face normal): 0→2→8→6, 1→5→7→3, center rotates in place.

_CW_GOES_TO  = [2, 5, 8, 1, 4, 7, 0, 3, 6]
_CCW_GOES_TO = [6, 3, 0, 7, 4, 1, 8, 5, 2]


def _face_rot(goes_to: list[int], r: Matrix) -> Matrix:
    blocks = [[_Z] * 9 for _ in range(9)]
    for src, dest in enumerate(goes_to):
        blocks[dest][src] = r
    return np.block(blocks)


def face_state() -> Matrix:
    """Return 9 cubies in solved orientation as a (27×3) state matrix."""
    return np.tile(np.eye(3, dtype=int), (9, 1))


FACE_TRANSFORMS: dict[str, Matrix] = {
    "f+": _face_rot(_CW_GOES_TO,  z_minus_90()),
    "f-": _face_rot(_CCW_GOES_TO, z_plus_90()),
    "u+": _face_rot(_CW_GOES_TO,  y_minus_90()),
    "u-": _face_rot(_CCW_GOES_TO, y_plus_90()),
    "r+": _face_rot(_CW_GOES_TO,  x_plus_90()),
    "r-": _face_rot(_CCW_GOES_TO, x_minus_90()),
}

TRANSFORMS.update(FACE_TRANSFORMS)
