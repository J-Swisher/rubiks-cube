import numpy as np
import pytest

from cube_kernel import (
    animation_payload,
    apply_sequence,
    apply_transform,
    identity,
    named_transforms,
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
