from datetime import datetime

def get_time_now() -> str:
    """
    Just return current local time.
    """
    return datetime.now()