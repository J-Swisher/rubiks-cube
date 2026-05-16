import numpy as np
import pytest

from cube_kernel import (
    animation_payload,
    apply_sequence,
    apply_transform,
    cycle_bwd,
    cycle_fwd,
    identity,
    initial_cubies_3,
    named_transforms,
    orient_at_0,
    orient_at_1,
    orient_at_2,
    swap_0_1,
    swap_0_2,
    swap_1_2,
    trace_sequence,
)
from cube_kernel.transforms import x_plus_90, z_plus_90


def test_identity_is_solved_state():
    assert np.array_equal(identity(), np.eye(3, dtype=int))


def test_apply_transform_left_multiplies_state():
    state = identity()
    transform = x_plus_90()

    assert np.array_equal(apply_transform(state, transform), transform @ state)


def test_apply_sequence_returns_only_the_final_state():
    transforms = [x_plus_90(), z_plus_90()]

    assert np.array_equal(
        apply_sequence(identity(), transforms),
        z_plus_90() @ x_plus_90() @ identity(),
    )


def test_trace_sequence_returns_start_and_every_intermediate_state():
    transforms = [x_plus_90(), z_plus_90()]
    states = trace_sequence(identity(), transforms)

    assert len(states) == 3
    assert np.array_equal(states[0], identity())
    assert np.array_equal(states[1], x_plus_90())
    assert np.array_equal(states[2], z_plus_90() @ x_plus_90())


def test_animation_payload_is_json_ready():
    payload = animation_payload(identity(), ["x+", "z+"])

    assert payload["transforms"] == ["x+", "z+"]
    assert len(payload["states"]) == 3
    assert payload["initial"] == identity().tolist()
    assert payload["final"] == (z_plus_90() @ x_plus_90()).tolist()


def test_named_transforms_reject_unknown_names():
    with pytest.raises(ValueError):
        named_transforms(["x+", "banana"])


# ── 3-cubie tests ─────────────────────────────────────────────────────────────

def _cubie(state, i):
    """Extract the 3×3 block for cubie i from a (9×3) state."""
    return state[3 * i : 3 * i + 3]


def test_initial_cubies_3_are_all_identity():
    s = initial_cubies_3()
    assert s.shape == (9, 3)
    for i in range(3):
        assert np.array_equal(_cubie(s, i), np.eye(3, dtype=int))


def test_orient_at_0_rotates_only_cubie_0():
    s = apply_transform(initial_cubies_3(), orient_at_0(x_plus_90()))
    assert np.array_equal(_cubie(s, 0), x_plus_90())
    assert np.array_equal(_cubie(s, 1), identity())
    assert np.array_equal(_cubie(s, 2), identity())


def test_orient_at_1_rotates_only_cubie_1():
    s = apply_transform(initial_cubies_3(), orient_at_1(x_plus_90()))
    assert np.array_equal(_cubie(s, 0), identity())
    assert np.array_equal(_cubie(s, 1), x_plus_90())
    assert np.array_equal(_cubie(s, 2), identity())


def test_orient_at_2_rotates_only_cubie_2():
    s = apply_transform(initial_cubies_3(), orient_at_2(x_plus_90()))
    assert np.array_equal(_cubie(s, 0), identity())
    assert np.array_equal(_cubie(s, 1), identity())
    assert np.array_equal(_cubie(s, 2), x_plus_90())


def test_swap_0_1_exchanges_cubies():
    s = apply_transform(initial_cubies_3(), orient_at_0(x_plus_90()))
    s = apply_transform(s, swap_0_1())
    assert np.array_equal(_cubie(s, 0), identity())
    assert np.array_equal(_cubie(s, 1), x_plus_90())
    assert np.array_equal(_cubie(s, 2), identity())


def test_swap_1_2_exchanges_cubies():
    s = apply_transform(initial_cubies_3(), orient_at_2(x_plus_90()))
    s = apply_transform(s, swap_1_2())
    assert np.array_equal(_cubie(s, 1), x_plus_90())
    assert np.array_equal(_cubie(s, 2), identity())


def test_swap_0_2_exchanges_cubies():
    s = apply_transform(initial_cubies_3(), orient_at_0(x_plus_90()))
    s = apply_transform(s, swap_0_2())
    assert np.array_equal(_cubie(s, 0), identity())
    assert np.array_equal(_cubie(s, 2), x_plus_90())


def test_cycle_fwd_advances_cubies():
    # put a marker on cubie 0, cycle forward → should land on slot 1
    s = apply_transform(initial_cubies_3(), orient_at_0(x_plus_90()))
    s = apply_transform(s, cycle_fwd())
    assert np.array_equal(_cubie(s, 0), identity())
    assert np.array_equal(_cubie(s, 1), x_plus_90())
    assert np.array_equal(_cubie(s, 2), identity())


def test_cycle_bwd_is_inverse_of_cycle_fwd():
    s = initial_cubies_3()
    s_fwd = apply_transform(s, cycle_fwd())
    s_roundtrip = apply_transform(s_fwd, cycle_bwd())
    assert np.array_equal(s_roundtrip, s)


def test_swap_0_1_is_its_own_inverse():
    s = apply_transform(initial_cubies_3(), orient_at_0(x_plus_90()))
    assert np.array_equal(apply_transform(apply_transform(s, swap_0_1()), swap_0_1()), s)
