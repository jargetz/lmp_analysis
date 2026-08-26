"""Regression tests for Plotly MapLibre maps and shareable report output."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app import generate_facility_report_html
from charts import add_chp_facility_traces, create_node_finder_map, create_pnode_map


ROOT = Path(__file__).resolve().parent


def _facility(**overrides):
    data = {
        "facility": "Plant A",
        "lat": 37.1,
        "lon": -121.1,
        "primary_sector": "Food processing",
        "county": "Santa Clara",
        "city": "San Jose",
        "cap_and_trade": "Yes",
        "total_ghg": 1_000.0,
        "co2": 900.0,
        "nox": 2.0,
        "sox": 1.0,
        "pm25": 0.5,
    }
    data.update(overrides)
    return data


def _nodes():
    return [
        {
            "pnode_id": "N1",
            "lat": 37.0,
            "lon": -121.0,
            "node_type": "PNode",
            "area": "Bay Area",
            "zone": "NP15",
            "avg_price": 10.0,
        },
        {
            "pnode_id": "N2",
            "lat": 34.0,
            "lon": -118.0,
            "node_type": "PNode",
            "area": "Los Angeles",
            "zone": "SP15",
            "avg_price": 20.0,
        },
    ]


def _nearest_node():
    return {
        "pnode_id": "N1",
        "lat": 37.0,
        "lon": -121.0,
        "node_type": "PNode",
        "zone": "NP15",
        "avg_price": 10.0,
    }


def _substation(name="Sub A", lat=37.2, lon=-121.2, dist_km=10.0):
    return {
        "substation_name": name,
        "lat": lat,
        "lon": lon,
        "owner": "PG&E",
        "highest_kv": "230 kV",
        "status": "Operational",
        "dist_km": dist_km,
    }


def _assert_maplibre_figure(fig, expected_style="carto-positron"):
    payload = fig.to_plotly_json()
    assert payload["data"]
    assert all(trace["type"] == "scattermap" for trace in payload["data"])
    assert payload["layout"]["map"]["style"] == expected_style
    assert "mapbox" not in payload["layout"]


def test_supported_maplibre_apis_are_available():
    assert hasattr(px, "scatter_map")
    assert hasattr(go, "Scattermap")


def test_pnode_map_works_without_deprecated_mapbox_aliases(monkeypatch):
    """Reproduce the production API surface that exposed the original failure."""
    monkeypatch.delattr(px, "scatter_mapbox", raising=False)

    fig = create_pnode_map(_nodes(), "B8", color_by="zone")

    _assert_maplibre_figure(fig)


def test_application_source_does_not_use_deprecated_mapbox_apis():
    source = (ROOT / "app.py").read_text() + (ROOT / "charts.py").read_text()
    for deprecated_name in ("scatter_mapbox", "Scattermapbox", "mapbox_style", "mapbox="):
        assert deprecated_name not in source


def test_pnode_zone_map_and_all_overlays_use_maplibre():
    facility = _facility()
    nearest_substation = _substation()
    nearest_lv_substation = _substation(
        name="Sub B", lat=37.15, lon=-121.15, dist_km=5.0
    )

    fig = create_pnode_map(
        _nodes(),
        "B8",
        color_by="zone",
        facilities=[facility],
        selected_facility=facility,
        nearest_node=_nearest_node(),
        nearest_substation=nearest_substation,
        nearest_lv_substation=nearest_lv_substation,
    )

    _assert_maplibre_figure(fig)
    assert fig.layout.map.center.lat == facility["lat"]
    assert fig.layout.map.center.lon == facility["lon"]
    assert fig.layout.map.zoom == 11
    assert {
        "Covered Facilities",
        "Selected Facility",
        "Nearest Node",
        "Nearest ≥110kV Substation",
        "Closer Lower-Voltage Substation",
    }.issubset({trace.name for trace in fig.data})


def test_pnode_price_map_uses_maplibre_color_scale():
    fig = create_pnode_map(_nodes(), "B8", color_by="price")

    _assert_maplibre_figure(fig)
    assert fig.layout.coloraxis.colorbar.title.text == "$/MWh"
    assert all(list(trace.lat) and list(trace.lon) for trace in fig.data)


def test_chp_overlay_uses_maplibre_trace():
    fig = go.Figure()
    chp = {
        **_facility(),
        "biomass_co2": 0.0,
        "non_biomass_ghg": 1_000.0,
    }

    add_chp_facility_traces(fig, [chp])

    assert len(fig.data) == 1
    assert fig.data[0].type == "scattermap"
    assert list(fig.data[0].lat) == [chp["lat"]]


def test_node_finder_map_uses_maplibre_for_every_layer():
    facility = {
        **_facility(),
        "fac_lat": 37.1,
        "fac_lon": -121.1,
        "node_b_avg": 10.0,
        "nearest_node": "N1",
        "node_zone": "NP15",
        "dist_km": 5.0,
        "node_lat": 37.0,
        "node_lon": -121.0,
    }
    communities = [{"name": "Community A", "lat": 37.2, "lon": -121.2}]

    fig = create_node_finder_map([facility], communities)

    _assert_maplibre_figure(fig)
    assert {trace.name for trace in fig.data} == {
        "Facilities (colored by nearest-node price)",
        "Nearest CAISO Nodes",
        "AB 617 Communities",
    }
    assert fig.layout.uirevision == "node_finder_map"


def test_shareable_report_embeds_maplibre_figure_with_carto_style():
    facility = _facility()
    nearest_substation = _substation()
    nearest_lv_substation = _substation(
        name="Sub B", lat=37.15, lon=-121.15, dist_km=5.0
    )
    substations = pd.DataFrame(
        [
            {
                "lat": 37.2,
                "lon": -121.2,
                "Substation_Name": "Sub A",
                "Owner": "PG&E",
                "Highest_kV": "230",
                "Status": "Operational",
            }
        ]
    )

    html = generate_facility_report_html(
        sel_facility=facility,
        all_facilities=[facility],
        node_to_analyze=_nearest_node(),
        node_price=10.0,
        dlap_name="DLAP_PGAE-APND",
        dlap_bx_avg=8.0,
        dlap_allhours=30.0,
        bx_label="B8",
        period_label="2025",
        nearest_substation=nearest_substation,
        nearest_lv_substation=nearest_lv_substation,
        substations_df=substations,
    )

    assert '"type":"scattermap"' in html
    assert '"style":"carto-positron"' in html
    assert '"scrollZoom": true' in html
    assert "scattermapbox" not in html.lower()
