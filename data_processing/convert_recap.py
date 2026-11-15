"""Compatibility entry point requiring episode outcomes by default."""

from .convert import CAMERAS, prepare_image
from .convert import convert as _convert
from .convert import main as _main

__all__ = ["CAMERAS", "convert", "main", "prepare_image"]


def convert(config, output):
    config = dict(config)
    config.setdefault("require_success", True)
    return _convert(config, output)


def main():
    _main(converter=convert)


if __name__ == "__main__":
    main()
