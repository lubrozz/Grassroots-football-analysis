def parse_timestamp(timestamp: str) -> float:
    """
    Parses a timestamp string in the format "HH:MM:SS" or "MM:SS" into seconds.

    Args:
        timestamp (str): The timestamp string to parse.

    Returns:
        float: The total number of seconds represented by the timestamp.

    Raises:
        ValueError: If the timestamp format is invalid.
    """
    parts = timestamp.strip().split(":")
    if not 2 <= len(parts) <= 3:
        raise ValueError(f"Expected mm:ss or hh:mm:ss format, got: {timestamp!r}")

    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        raise ValueError(
            f"Timestamp contains non-numeric values: {timestamp!r}"
        ) from None

    *hours_and_minutes, seconds = numbers
    if seconds < 0 or seconds >= 60 or any(n < 0 for n in hours_and_minutes):
        raise ValueError(f"Seconds must be in the range [0, 60), got: {seconds!r}")

    total = 0.0
    for number in hours_and_minutes:
        total = total * 60 + number
    return total * 60 + seconds


def format_timestamp(seconds: float) -> str:
    """
    Formats a number of seconds into a timestamp string in the format "HH:MM:SS" or "MM:SS" for display.

    Args:
        seconds (float): The total number of seconds to format.

    Returns:
        str: The formatted timestamp string.
    """
    whole = int(seconds)
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
