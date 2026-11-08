"""List config values that are still unmeasured.

    python -m tools.check_config [path/to/fusion.yaml]
"""
import sys

from fusion.config import load_config, unmeasured


def main():
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    missing = unmeasured(cfg)
    if not missing:
        print("All values filled in.")
        return
    print(f"{len(missing)} values not measured yet:")
    for k in missing:
        print("  -", k)


if __name__ == "__main__":
    main()
