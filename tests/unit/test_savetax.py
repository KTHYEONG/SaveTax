"""Unit tests for savetax package."""

from src.savetax import __version__


def test_package_version() -> None:
    """Verify package version is defined."""
    assert __version__ == "0.1.0"
