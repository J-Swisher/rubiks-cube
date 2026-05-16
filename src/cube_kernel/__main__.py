"""Print a tiny sample animation payload as JSON."""

from __future__ import annotations

import json

from cube_kernel.api import animation_payload, identity


def main() -> None:
    payload = animation_payload(identity(), ["x+", "z+", "y+"])
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
