"""Regression tests for the Site Analysis facility search."""

from app import _facility_search_matches


FACILITIES = [
    {
        "facility": "E & J Gallo Winery - Fresno",
        "city": "Fresno",
        "county": "Fresno",
    },
    {
        "facility": "Gallo Glass Company",
        "city": "Modesto",
        "county": "Stanislaus",
    },
    {
        "facility": "E & J Gallo Winery - Livingston",
        "city": "Livingston",
        "county": "Merced",
    },
]


def test_exact_gallo_glass_search_is_first_result():
    results = _facility_search_matches(FACILITIES, "Gallo Glass Company")
    assert [row["facility"] for row in results] == ["Gallo Glass Company"]


def test_search_is_case_insensitive_and_matches_location():
    assert len(_facility_search_matches(FACILITIES, "gallo")) == 3
    results = _facility_search_matches(FACILITIES, "stanislaus")
    assert [row["facility"] for row in results] == ["Gallo Glass Company"]


def test_blank_or_unknown_search_has_no_results():
    assert _facility_search_matches(FACILITIES, "   ") == []
    assert _facility_search_matches(FACILITIES, "not a real facility") == []
