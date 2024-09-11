import os

from dynaconf import Dynaconf, Validator

# Determine if we're running tests
is_testing = os.environ.get("PYTEST_CURRENT_TEST") is not None

settings = Dynaconf(
    envvar_prefix="MEM0RYLOL",
    settings_files=[
        "config/.secrets.toml",
        "config/settings.toml",
    ],
)

# Ensure required settings are present
settings.setdefault("LANCEDB_CONNECTION_STRING", "./lancedb_data")

# Validate configuration
settings.validators.register(Validator("LANCEDB_CONNECTION_STRING", must_exist=True))

# Validate all registered validators
settings.validators.validate()
