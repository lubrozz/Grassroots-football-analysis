import pytest
from src.utils.time_utils import format_timestamp, parse_timestamp


@pytest.mark.parametrize(
    ("timestamp", "expected"),
    [
        ("12:43", 763.0),
        ("0:05", 5.0),
        ("1:02:05", 3725.0),
        ("12:43.5", 763.5),
        (" 3:00 ", 180.0),
    ],
)
def test_parse_timestamp(timestamp: str, expected: float) -> None:
    assert parse_timestamp(timestamp) == expected


@pytest.mark.parametrize("timestamp", ["743", "12:60", "ab:cd", "1:2:3:4", "-1:30"])
def test_parse_timestamp_rejects_invalid(timestamp: str) -> None:
    with pytest.raises(ValueError):
        parse_timestamp(timestamp)


def test_format_round_trip() -> None:
    assert format_timestamp(763.0) == "12:43"
    assert format_timestamp(3725.0) == "1:02:05"
    assert parse_timestamp(format_timestamp(763.0)) == 763.0
