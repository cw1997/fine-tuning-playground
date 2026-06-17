"""
Environment fixes for common conda / Windows SSL misconfiguration.

Some conda activate scripts set SSL_CERT_FILE to a path that does not exist,
which breaks httpx and Hugging Face Hub downloads.
"""

import os

import certifi

_SSL_ENV_VARS = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")


def fix_ssl_cert_env() -> None:
    """Point broken SSL env vars at certifi's CA bundle."""
    cert_path = certifi.where()
    for var in _SSL_ENV_VARS:
        value = os.environ.get(var)
        if value and not os.path.isfile(value):
            os.environ[var] = cert_path
            print(f"  Note: {var} pointed to missing file; using certifi bundle.")
