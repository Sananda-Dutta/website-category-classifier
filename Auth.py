# ═══════════════════════════════════════════════════════════════════════════════
# auth.py — API credential verification
# Website Category Classifier  v2.7.0
#
# NOTE: Primary auth is handled in api.py via the RAPIDAPI_PROXY_SECRET
# middleware (X-RapidAPI-Proxy-Secret header check).
#
# This module exists for optional secondary credential checks —
# e.g. direct API key validation for non-RapidAPI clients.
# ═══════════════════════════════════════════════════════════════════════════════

import os
from fastapi import Request, HTTPException


# ─────────────────────────────────────────────
# Optional direct API key (for non-RapidAPI access)
# Set DIRECT_API_KEY env var in Render to enable.
# Leave unset to allow unauthenticated direct access.
# ─────────────────────────────────────────────
DIRECT_API_KEY = os.getenv("DIRECT_API_KEY", "").strip()


def verify_api_credentials(request: Request) -> bool:
    """
    Secondary credential check for direct (non-RapidAPI) clients.

    Returns True if:
    - No DIRECT_API_KEY is configured (open access)
    - X-API-Key header matches the configured DIRECT_API_KEY

    Raises HTTPException 401 if key is wrong.

    Usage in endpoints (optional — primary auth is middleware):
        from auth import verify_api_credentials
        verify_api_credentials(request)
    """
    if not DIRECT_API_KEY:
        # No key configured — allow all direct requests
        return True

    incoming_key = request.headers.get("X-API-Key", "").strip()

    if not incoming_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Missing API key.",
                "hint":  "Pass your key in the X-API-Key header, "
                         "or access via RapidAPI at rapidapi.com."
            }
        )

    if incoming_key != DIRECT_API_KEY:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Invalid API key.",
                "hint":  "Check your key or access via RapidAPI at rapidapi.com."
            }
        )

    return True