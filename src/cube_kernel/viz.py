"""3D visualization for cubie orientation states.

The core primitive is ``plot_cubie``, which renders a single 3×3 orientation
matrix as an interactive colored 3D box.  ``plot_cubie_array`` composes any
number of cubies arranged in an arbitrary N-dimensional grid.

Input format for ``plot_cubie_array``: an ndarray of shape ``(*grid_shape, 3, 3)``
where the last two dimensions are always the 3×3 orientation matrix and the
leading dimensions define the grid arrangement.  Examples:

    (3, 3)               — single cubie        (0-D grid)
    (N, 3, 3)            — 1-D row of N cubies
    (N, M, 3, 3)         — 2-D N×M grid        (e.g. a face)
    (N, M, P, 3, 3)      — 3-D N×M×P grid      (e.g. a full 3×3×3 cube)
    (N, M, P, Q, 3, 3)   — 4-D hypercube        (tiled into 3-D space)

All-zero (3, 3) blocks are treated as empty slots and skipped.

Dimension-to-world-axis mapping cycles through y (inverted), x, z:
    dim 0 → −y  (row; index 0 = top)
    dim 1 → +x  (column; index 0 = left)
    dim 2 → +z  (depth; index 0 = back)
    dim 3 → −y  (cluster row, larger scale)
    dim 4 → +x  (cluster column, larger scale)
    …

Face colors are local-axis colors (fixed to the cubie's frame):

    +z = Red  (front)   −z = Orange (back)
    +y = White (top)    −y = Yellow (bottom)
    +x = Green (right)  −x = Blue   (left)

Requires ``plotly``.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

Matrix = NDArray[np.floating]

# ── face colors (keyed by local-axis direction tuple) ─────────────────────────

AXIS_FACE_COLORS: dict[tuple[int, int, int], str] = {
    ( 0,  0,  1): "#B71234",  # +z  Front  Red
    ( 0,  0, -1): "#FF5800",  # -z  Back   Orange
    ( 0,  1,  0): "#FFFFFF",  # +y  Up     White
    ( 0, -1,  0): "#FFD500",  # -y  Down   Yellow
    ( 1,  0,  0): "#009B48",  # +x  Right  Green
    (-1,  0,  0): "#0046AD",  # -x  Left   Blue
}

# ── unit-cube geometry ────────────────────────────────────────────────────────
#
# 8 vertices at (±0.5, ±0.5, ±0.5):
#   0 = (−,−,−)  1 = (+,−,−)  2 = (+,+,−)  3 = (−,+,−)
#   4 = (−,−,+)  5 = (+,−,+)  6 = (+,+,+)  7 = (−,+,+)

_VERTS = np.array(
    [
        [-0.5, -0.5, -0.5],  # 0
        [ 0.5, -0.5, -0.5],  # 1
        [ 0.5,  0.5, -0.5],  # 2
        [-0.5,  0.5, -0.5],  # 3
        [-0.5, -0.5,  0.5],  # 4
        [ 0.5, -0.5,  0.5],  # 5
        [ 0.5,  0.5,  0.5],  # 6
        [-0.5,  0.5,  0.5],  # 7
    ],
    dtype=float,
)

# 12 triangles — 2 per face, order: +z, −z, +y, −y, +x, −x
_TRI_I = [4, 4,  0, 0,  3, 3,  0, 0,  1, 1,  0, 0]
_TRI_J = [5, 6,  1, 2,  2, 6,  1, 5,  2, 6,  3, 7]
_TRI_K = [6, 7,  2, 3,  6, 7,  5, 4,  6, 5,  7, 4]

_TRI_COLORS: list[str] = (
    [AXIS_FACE_COLORS[( 0,  0,  1)]] * 2   # +z
    + [AXIS_FACE_COLORS[( 0,  0, -1)]] * 2  # -z
    + [AXIS_FACE_COLORS[( 0,  1,  0)]] * 2  # +y
    + [AXIS_FACE_COLORS[( 0, -1,  0)]] * 2  # -y
    + [AXIS_FACE_COLORS[( 1,  0,  0)]] * 2  # +x
    + [AXIS_FACE_COLORS[(-1,  0,  0)]] * 2  # -x
)

# Orientation-arrow colors (x=dark-red, y=dark-green, z=dark-blue)
_ARROW_COLORS = ["#CC0000", "#00AA00", "#0000CC"]


# ── internal helpers ──────────────────────────────────────────────────────────

def _grid_to_world(
    idx: tuple[int, ...],
    shape: tuple[int, ...],
    spacing: float,
    cluster_gap: float,
) -> tuple[float, float, float]:
    """Map a D-dimensional grid index to a 3D world position.

    Dimensions cycle through world axes in the order: −y (row, 0=top),
    +x (col, 0=left), +z (depth, 0=back).  Dimensions 3, 4, 5 reuse the
    same axis cycle but at a larger scale so that each higher-tier index
    creates a new cluster of lower-tier structures.

    The unit spacing for tier-1 dimensions (dim 3, 4, 5) is derived from
    the extent of the corresponding tier-0 dimension plus ``cluster_gap``
    extra spacings of clearance.
    """
    D = len(idx)
    # Pre-compute the unit distance for each dimension.
    units: list[float] = [spacing] * D
    for d in range(3, D):
        parent = d - 3
        parent_extent = (shape[parent] - 1) * units[parent]
        units[d] = parent_extent + cluster_gap * spacing

    wx = wy = wz = 0.0
    for d, (i, n, u) in enumerate(zip(idx, shape, units)):
        center = (n - 1) / 2.0
        offset = (i - center) * u
        axis = d % 3
        if axis == 0:
            wy -= offset  # row 0 at top (+y)
        elif axis == 1:
            wx += offset
        else:
            wz += offset

    return wx, wy, wz


# ── public API ────────────────────────────────────────────────────────────────

def plot_cubie(
    O: np.ndarray,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    size: float = 1.0,
    show_axes: bool = False,
) -> list[go.BaseTraceType]:
    """Return Plotly traces for one cubie.

    Parameters
    ----------
    O:
        3×3 orientation matrix.  Identity = solved (aligned with world axes).
        All 8 vertices of the unit box are rotated by O before translating to
        *center*, so the colored faces reflect the current orientation visually.
    center:
        World-space position of the cubie's center.
    size:
        Edge length of the cube (default 1.0).
    show_axes:
        If True, add three colored arrows showing the local x (dark-red),
        y (dark-green), z (dark-blue) axes in their current world directions.

    Returns
    -------
    list of Plotly trace objects (Mesh3d + optional Scatter3d).
    """
    O = np.asarray(O, dtype=float)
    c = np.asarray(center, dtype=float)

    verts = (O @ (_VERTS * size).T).T + c

    traces: list[go.BaseTraceType] = [
        go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=_TRI_I,
            j=_TRI_J,
            k=_TRI_K,
            facecolor=_TRI_COLORS,
            flatshading=True,
            showscale=False,
            hoverinfo="none",
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
        )
    ]

    if show_axes:
        for axis_idx in range(3):
            tip = c + 0.75 * size * O[:, axis_idx]
            traces.append(
                go.Scatter3d(
                    x=[c[0], tip[0]],
                    y=[c[1], tip[1]],
                    z=[c[2], tip[2]],
                    mode="lines",
                    line=dict(color=_ARROW_COLORS[axis_idx], width=6),
                    showlegend=False,
                    hoverinfo="none",
                )
            )

    return traces


def parse_cubie_array(
    matrix: np.ndarray,
) -> dict[tuple[int, ...], np.ndarray]:
    """Parse a (*grid_shape, 3, 3) tensor into a dict of cubie positions.

    The last two dimensions must be (3, 3) — the orientation matrix.  All
    leading dimensions form the grid shape.  A plain (3, 3) array is treated
    as a 0-D grid containing a single cubie.

    All-zero (3, 3) blocks are empty slots and are omitted from the result.

    Returns
    -------
    dict mapping grid_index_tuple → 3×3 orientation array.

    Raises
    ------
    ValueError if the last two dimensions are not (3, 3).
    """
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim < 2 or arr.shape[-2:] != (3, 3):
        raise ValueError(
            f"last two dimensions must be (3, 3); got shape {arr.shape}"
        )
    grid_shape = arr.shape[:-2]
    index_iter: Iterable[tuple[int, ...]] = (
        np.ndindex(*grid_shape) if grid_shape else [()]
    )
    cubies: dict[tuple[int, ...], np.ndarray] = {}
    for idx in index_iter:
        block: np.ndarray = arr[idx] if idx else arr
        if not np.all(block == 0):
            cubies[idx] = block
    return cubies


def plot_cubie_array(
    matrix: np.ndarray,
    spacing: float = 1.1,
    show_axes: bool = False,
    title: str = "",
    cluster_gap: float = 2.0,
) -> go.Figure:
    """Render a (*grid_shape, 3, 3) cubie tensor as an interactive 3D figure.

    Parameters
    ----------
    matrix:
        ndarray of shape (*grid_shape, 3, 3).  The last two dims are always
        the 3×3 orientation matrix; the leading dims define the grid.

        - (3, 3)              — single cubie
        - (N, 3, 3)           — 1-D row of N cubies
        - (N, M, 3, 3)        — 2-D N×M grid  (e.g. a face)
        - (N, M, P, 3, 3)     — 3-D N×M×P cube
        - (N, M, P, Q, 3, 3)  — 4-D hypercube (projected into 3-D space)

        All-zero (3, 3) blocks are treated as empty slots.
    spacing:
        Center-to-center distance between adjacent cubies within a tier.
    show_axes:
        Draw local-axis orientation arrows on each cubie.  Recommended only
        for small arrays.
    title:
        Optional figure title.
    cluster_gap:
        Clearance between clusters at higher tiers, as a multiple of
        ``spacing``.  Increase for better separation in 4-D+ arrangements.

    Returns
    -------
    go.Figure — call ``.show()`` in a Jupyter notebook to display.
    """
    arr = np.asarray(matrix, dtype=float)
    cubies = parse_cubie_array(arr)
    if not cubies:
        return go.Figure()

    grid_shape = arr.shape[:-2]

    traces: list[go.BaseTraceType] = []
    for idx, O in cubies.items():
        cx, cy, cz = _grid_to_world(idx, grid_shape, spacing, cluster_gap)
        traces.extend(plot_cubie(O, center=(cx, cy, cz), show_axes=show_axes))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)) if title else None,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            aspectmode="data",
            bgcolor="#f8f8f8",
        ),
        paper_bgcolor="#f8f8f8",
        margin=dict(l=0, r=0, t=40 if title else 0, b=0),
    )
    return fig

