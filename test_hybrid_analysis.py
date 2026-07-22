"""Unit tests for hybrid BX dispatch calculations (no database required)."""

import pytest
import duckdb

from hybrid_analysis import (
    HybridAssumptions,
    chp_extraction_case,
    natural_gas_case,
    run_hybrid_dispatch,
    sensitivity_matrix,
)
from subprocess_query import get_hybrid_bx_hours


@pytest.fixture
def bx_rows():
    # Four selected BX hours on each of two illustrative days.
    return [
        {"date": "2024-01-01", "hour": hour, "lmp": price}
        for hour, price in enumerate([0.0, 10.0, 20.0, 30.0], start=1)
    ] + [
        {"date": "2024-01-02", "hour": hour, "lmp": price}
        for hour, price in enumerate([40.0, 50.0, 60.0, 70.0], start=1)
    ]


def test_supplied_gas_components_reproduce_total_and_emissions():
    gas = natural_gas_case()
    assert gas["total_per_mwh_th"] == pytest.approx(33.33, abs=0.03)
    assert gas["carbon_per_mwh_th"] == pytest.approx(5.94, abs=0.03)
    assert gas["emissions_ton_per_mwh_th"] == pytest.approx(0.2128, abs=0.0002)


def test_fixed_hour_share_drops_highest_observations(bx_rows):
    assumptions = HybridAssumptions(
        electric_performance=1.0,
        nbc_per_mwh_th=0.0,
        tac_per_mwh_th=0.0,
        backup_cost_per_mwh_th=25.0,
    )
    result, summary = run_hybrid_dispatch(
        bx_rows, assumptions, mode="fixed_hours", backup_share_percent=25.0,
    )
    dropped = sorted(result.loc[result["use_backup"], "lmp"].tolist())
    assert dropped == [60.0, 70.0]
    assert summary["backup_share_percent"] == pytest.approx(25.0)


def test_fixed_day_share_drops_complete_highest_price_day(bx_rows):
    assumptions = HybridAssumptions(
        electric_performance=1.0,
        nbc_per_mwh_th=0.0,
        tac_per_mwh_th=0.0,
        backup_cost_per_mwh_th=25.0,
    )
    result, summary = run_hybrid_dispatch(
        bx_rows, assumptions, mode="fixed_days", backup_share_percent=50.0,
    )
    dropped_dates = set(result.loc[result["use_backup"], "date"].astype(str))
    assert dropped_dates == {"2024-01-02"}
    assert summary["backup_hours"] == 4
    assert summary["backup_days"] == 1


def test_economic_dispatch_uses_adjusted_useful_heat_cost(bx_rows):
    assumptions = HybridAssumptions(
        electric_performance=0.95,
        nbc_per_mwh_th=0.0,
        tac_per_mwh_th=0.0,
        backup_cost_per_mwh_th=33.33,
    )
    result, summary = run_hybrid_dispatch(bx_rows, assumptions, mode="economic")
    assert summary["break_even_lmp"] == pytest.approx(31.6635, abs=0.001)
    assert result.loc[result["use_backup"], "lmp"].min() == 40.0


def test_current_nbc_and_tac_make_break_even_negative(bx_rows):
    assumptions = HybridAssumptions(
        electric_performance=0.95,
        nbc_per_mwh_th=33.0,
        tac_per_mwh_th=16.39,
        backup_cost_per_mwh_th=33.33,
    )
    _, summary = run_hybrid_dispatch(bx_rows, assumptions, mode="economic")
    assert summary["break_even_lmp"] == pytest.approx(-15.257, abs=0.002)


def test_chp_three_to_one_outputs_sum_to_95_percent():
    chp = chp_extraction_case(
        fuel_cost_per_mwh_in=28.33,
        total_useful_efficiency=0.95,
        heat_to_power_ratio=3.0,
        electricity_value_per_mwh=50.0,
    )
    assert chp["heat_yield_per_mwh_in"] == pytest.approx(0.7125)
    assert chp["power_yield_per_mwh_in"] == pytest.approx(0.2375)
    assert chp["electricity_mwh_per_mwh_th"] == pytest.approx(1 / 3)
    assert chp["net_cost_per_mwh_th"] < chp["gross_cost_per_mwh_th"]


def test_sensitivity_returns_every_grid_cell(bx_rows):
    assumptions = HybridAssumptions(
        electric_performance=0.95,
        nbc_per_mwh_th=0.0,
        tac_per_mwh_th=0.0,
    )
    result = sensitivity_matrix(
        bx_rows,
        assumptions,
        citygate_values=[2.0, 4.0],
        carbon_price_values=[0.0, 50.0, 100.0],
    )
    assert len(result) == 6
    assert set(result.columns) == {
        "citygate_per_mmbtu",
        "carbon_price_per_ton",
        "backup_share_percent",
        "blended_cost_per_mwh_th",
        "break_even_lmp",
    }


def test_hybrid_query_selects_b_cheapest_hours_within_each_day():
    conn = duckdb.connect(":memory:")
    conn.execute("""
        CREATE TABLE node_hourly_lmp (
            opr_dt DATE,
            node VARCHAR,
            opr_hr INTEGER,
            mw DOUBLE
        )
    """)
    # Verify the local fixture schema before exercising the production query.
    columns = {row[0] for row in conn.execute("DESCRIBE node_hourly_lmp").fetchall()}
    assert columns == {"opr_dt", "node", "opr_hr", "mw"}
    for day, offset in [("2024-01-01", 0), ("2024-01-02", 100)]:
        conn.executemany(
            "INSERT INTO node_hourly_lmp VALUES (?, 'TEST_NODE', ?, ?)",
            [(day, hour, float(offset + hour)) for hour in range(1, 25)],
        )
    result = get_hybrid_bx_hours(conn, "TEST_NODE", 4, "2024")
    conn.close()

    assert result["success"] is True
    assert result["day_count"] == 2
    assert len(result["observations"]) == 8
    by_day = {}
    for row in result["observations"]:
        by_day.setdefault(row["date"], []).append(row["hour"])
    assert all(sorted(hours) == [1, 2, 3, 4] for hours in by_day.values())
