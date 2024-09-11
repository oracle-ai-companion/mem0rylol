from dynaconf import Dynaconf

settings = Dynaconf(
    envvar_prefix="MEM0RYLOL",
    settings_files=['config/.secrets.toml', 'config/settings.toml'],
)

# Ensure the necessary settings are defined
settings.lancedb_connection_string
settings.google_genai_api_key
settings.cerebras_api_key
