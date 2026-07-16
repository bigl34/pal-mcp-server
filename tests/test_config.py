"""
Tests for configuration
"""

from config import (
    DEFAULT_MODEL,
    TEMPERATURE_ANALYTICAL,
    TEMPERATURE_BALANCED,
    TEMPERATURE_CREATIVE,
    __author__,
    __updated__,
    __version__,
)


class TestConfig:
    """Test configuration values"""

    def test_version_info(self):
        """Test version information exists and has correct format"""
        # Check public version format with optional PEP 440 local fork tag.
        assert isinstance(__version__, str)
        public_version, *local_tag = __version__.split("+", 1)
        assert len(public_version.split(".")) == 3  # Major.Minor.Patch
        if local_tag:
            assert local_tag[0].startswith("fork.")

        # Check author
        assert __author__ == "Fahad Gilani"

        # Check updated date exists (don't assert on specific format/value)
        assert isinstance(__updated__, str)

    def test_model_config(self):
        """Test model configuration"""
        # DEFAULT_MODEL is set in conftest.py for tests
        assert DEFAULT_MODEL == "gemini-2.5-flash"

    def test_temperature_defaults(self):
        """Test temperature constants"""
        assert TEMPERATURE_ANALYTICAL == 1.0
        assert TEMPERATURE_BALANCED == 1.0
        assert TEMPERATURE_CREATIVE == 1.0
