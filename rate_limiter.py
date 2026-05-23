from slowapi import Limiter
from slowapi.util import get_remote_address

# Limit each IP to 30 requests per minute
limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])