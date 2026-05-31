"""Frontend-facing functions for cubie orientation state.

Single-cubie state: 3x3 orientation matrix (identity = solved).
Three-cubie state: 9x3 matrix — cubie k occupies rows 3k..3k+2.

In both cases, applying a transform means left-multiplying the state.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cube_kernel.transforms import TRANSFORMS

Matrix = NDArray[np.integer]
CubieArray = NDArray[np.integer]  # shape (N, 3, 3)


def identity() -> Matrix:
    """Return the solved orientation state."""

    return np.eye(3, dtype=int)


def apply_transform(state: Matrix, transform: Matrix) -> Matrix:
    """Apply one transform to one state."""

    return np.asarray(transform, dtype=int) @ np.asarray(state, dtype=int)


def apply_sequence(state: Matrix, transforms: list[Matrix]) -> Matrix:
    """Apply transforms in order and return only the final state."""

    current = np.asarray(state, dtype=int)
    for transform in transforms:
        current = apply_transform(current, transform)
    return current


def trace_sequence(state: Matrix, transforms: list[Matrix]) -> list[Matrix]:
    """Apply transforms in order and return every state, including start."""

    current = np.asarray(state, dtype=int)
    states = [current]

    for transform in transforms:
        current = apply_transform(current, transform)
        states.append(current)

    return states


def named_transform(name: str) -> Matrix:
    """Look up a transform by its short API name."""

    try:
        return TRANSFORMS[name]
    except KeyError as error:
        allowed = ", ".join(sorted(TRANSFORMS))
        raise ValueError(f"unknown transform {name!r}; allowed: {allowed}") from error


def named_transforms(names: list[str]) -> list[Matrix]:
    """Look up several transforms by name."""

    return [named_transform(name) for name in names]


def animation_payload(state: Matrix, transform_names: list[str]) -> dict[str, object]:
    """Return JSON-ready data for a frontend animation."""

    transforms = named_transforms(transform_names)
    states = trace_sequence(state, transforms)

    return {
        "initial": matrix_to_list(states[0]),
        "transforms": list(transform_names),
        "states": [matrix_to_list(next_state) for next_state in states],
        "final": matrix_to_list(states[-1]),
    }


def matrix_to_list(matrix: Matrix) -> list[list[int]]:
    """Convert a NumPy matrix to plain Python lists for JSON."""

    return [[int(value) for value in row] for row in np.asarray(matrix, dtype=int)]


def initial_cubies_3() -> Matrix:
    """Return 3 cubies in solved orientation as a (9x3) state matrix."""

    return np.tile(np.eye(3, dtype=int), (3, 1))
