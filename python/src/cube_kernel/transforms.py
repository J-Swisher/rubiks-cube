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
