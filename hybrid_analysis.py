"""Pure calculation helpers for BX hybrid-operation scenarios.

The database query identifies the B cheapest hours within each day.  This
module applies a second-stage dispatch decision to those selected BX hours.
Keeping these functions independent of Streamlit makes the methodology easy
to test and lets sensitivity analysis reuse one MotherDuck result.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Literal

import pandas as pd


DispatchMode = Literal["fixed_hours", "fixed_days", "economic"]


@dataclass(frozen=True)
class HybridAssumptions:
    """Economic assumptions expressed on consistent energy bases."""

    electric_performance: float = 0.95  # useful MWh-th per MWh-e
    nbc_per_mwh_th: float = 33.0
    tac_per_mwh_th: float = 16.39
    procurement_per_mwh_e: float = 0.0
    electric_ghg_per_mwh_e: float = 0.0
    electric_value_per_mwh_th: float = 0.0
    backup_cost_per_mwh_th: float = 33.33
    backup_value_per_mwh_th: float = 0.0

    def validate(self) -> None:
        if not isfinite(self.electric_performance) or self.electric_performance <= 0:
            raise ValueError("Electric performance must be greater than zero")
        for name, value in self.__dict__.items():
            if name == "electric_performance":
                continue
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")


def natural_gas_case(
    *,
    citygate_per_mmbtu_in: float = 3.65,
    transportation_per_mmbtu_in: float = 2.51,
    pppc_per_mmbtu_in: float = 0.66,
    carbon_price_per_ton: float = 27.92,
    emissions_ton_per_therm: float = 0.0053,
    gas_efficiency: float = 0.85,
) -> dict[str, float]:
    """Return gas costs and emissions per useful MWh-th.

    One MWh equals 3.412141633 MMBtu and one MMBtu equals 10 therms.
    """
    if gas_efficiency <= 0:
        raise ValueError("Gas efficiency must be greater than zero")
    mmbtu_per_mwh = 3.412141633
    emissions_per_mmbtu = emissions_ton_per_therm * 10.0
    noncarbon_input = (
        citygate_per_mmbtu_in
        + transportation_per_mmbtu_in
        + pppc_per_mmbtu_in
    )
    carbon_input = carbon_price_per_ton * emissions_per_mmbtu
    input_cost_per_mwh = (noncarbon_input + carbon_input) * mmbtu_per_mwh
    emissions_per_mwh_th = emissions_per_mmbtu * mmbtu_per_mwh / gas_efficiency
    return {
        "noncarbon_per_mwh_th": noncarbon_input * mmbtu_per_mwh / gas_efficiency,
        "carbon_per_mwh_th": carbon_input * mmbtu_per_mwh / gas_efficiency,
        "total_per_mwh_th": input_cost_per_mwh / gas_efficiency,
        "emissions_ton_per_mwh_th": emissions_per_mwh_th,
        "input_cost_per_mmbtu": noncarbon_input + carbon_input,
    }


def chp_extraction_case(
    *,
    fuel_cost_per_mwh_in: float,
    total_useful_efficiency: float = 0.95,
    heat_to_power_ratio: float = 3.0,
    electricity_value_per_mwh: float = 0.0,
) -> dict[str, float]:
    """Value CHP extraction output on a one-useful-MWh-th basis."""
    if total_useful_efficiency <= 0 or heat_to_power_ratio <= 0:
        raise ValueError("CHP efficiency and heat-to-power ratio must be positive")
    heat_yield = total_useful_efficiency * heat_to_power_ratio / (heat_to_power_ratio + 1.0)
    power_yield = total_useful_efficiency / (heat_to_power_ratio + 1.0)
    fuel_needed = 1.0 / heat_yield
    electricity_generated = power_yield * fuel_needed
    gross_cost = fuel_cost_per_mwh_in * fuel_needed
    electricity_credit = electricity_generated * electricity_value_per_mwh
    return {
        "heat_yield_per_mwh_in": heat_yield,
        "power_yield_per_mwh_in": power_yield,
        "fuel_mwh_in_per_mwh_th": fuel_needed,
        "electricity_mwh_per_mwh_th": electricity_generated,
        "gross_cost_per_mwh_th": gross_cost,
        "electricity_credit_per_mwh_th": electricity_credit,
        "net_cost_per_mwh_th": gross_cost - electricity_credit,
    }


def _normalize_hours(rows: Iterable[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows).copy()
    required = {"date", "hour", "lmp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Hybrid BX data missing columns: {', '.join(sorted(missing))}")
    if df.empty:
        raise ValueError("No BX hourly observations were returned")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["hour"] = pd.to_numeric(df["hour"], errors="raise").astype(int)
    df["lmp"] = pd.to_numeric(df["lmp"], errors="raise").astype(float)
    return df.sort_values(["date", "hour"]).reset_index(drop=True)


def run_hybrid_dispatch(
    rows: Iterable[dict],
    assumptions: HybridAssumptions,
    *,
    mode: DispatchMode = "fixed_hours",
    backup_share_percent: float = 15.0,
) -> tuple[pd.DataFrame, dict[str, float | int | None]]:
    """Apply a hybrid dispatch rule after daily BX hours have been selected."""
    assumptions.validate()
    if not 0 <= backup_share_percent <= 100:
        raise ValueError("Backup share must be between zero and 100")

    df = _normalize_hours(rows)
    df["electric_cost_per_mwh_th"] = (
        (df["lmp"] + assumptions.procurement_per_mwh_e + assumptions.electric_ghg_per_mwh_e)
        / assumptions.electric_performance
        + assumptions.nbc_per_mwh_th
        + assumptions.tac_per_mwh_th
        - assumptions.electric_value_per_mwh_th
    )
    backup_net_cost = assumptions.backup_cost_per_mwh_th - assumptions.backup_value_per_mwh_th

    df["use_backup"] = False
    if mode == "fixed_hours":
        drop_count = int(round(len(df) * backup_share_percent / 100.0))
        if drop_count:
            idx = df.sort_values(["lmp", "date", "hour"], ascending=[False, True, True]).head(drop_count).index
            df.loc[idx, "use_backup"] = True
    elif mode == "fixed_days":
        daily = df.groupby("date", as_index=False)["lmp"].mean()
        drop_days = int(round(len(daily) * backup_share_percent / 100.0))
        if drop_days:
            selected_days = set(daily.nlargest(drop_days, "lmp")["date"])
            df["use_backup"] = df["date"].isin(selected_days)
    elif mode == "economic":
        df["use_backup"] = df["electric_cost_per_mwh_th"] > backup_net_cost
    else:
        raise ValueError(f"Unknown dispatch mode: {mode}")

    df["dispatch"] = df["use_backup"].map({True: "Backup", False: "Electric"})
    df["selected_cost_per_mwh_th"] = df["electric_cost_per_mwh_th"]
    df.loc[df["use_backup"], "selected_cost_per_mwh_th"] = backup_net_cost

    electric = df.loc[~df["use_backup"]]
    backup = df.loc[df["use_backup"]]
    all_electric_cost = float(df["electric_cost_per_mwh_th"].mean())
    blended_cost = float(df["selected_cost_per_mwh_th"].mean())
    backup_share = float(df["use_backup"].mean() * 100.0)
    evening = df["hour"].between(17, 21)
    dropped_evening = backup["hour"].between(17, 21) if not backup.empty else pd.Series(dtype=bool)

    break_even_lmp = (
        (backup_net_cost + assumptions.electric_value_per_mwh_th
         - assumptions.nbc_per_mwh_th - assumptions.tac_per_mwh_th)
        * assumptions.electric_performance
        - assumptions.procurement_per_mwh_e
        - assumptions.electric_ghg_per_mwh_e
    )
    cutoff_lmp = None
    if not backup.empty and mode != "fixed_days":
        cutoff_lmp = float(backup["lmp"].min())

    summary: dict[str, float | int | None] = {
        "observation_count": int(len(df)),
        "day_count": int(df["date"].nunique()),
        "standard_bx_lmp": float(df["lmp"].mean()),
        "retained_electric_lmp": float(electric["lmp"].mean()) if not electric.empty else None,
        "backup_share_percent": backup_share,
        "backup_hours": int(df["use_backup"].sum()),
        "backup_days": int(backup["date"].nunique()),
        "blended_cost_per_mwh_th": blended_cost,
        "all_electric_cost_per_mwh_th": all_electric_cost,
        "all_backup_cost_per_mwh_th": backup_net_cost,
        "savings_vs_all_electric": all_electric_cost - blended_cost,
        "savings_vs_all_backup": backup_net_cost - blended_cost,
        "break_even_lmp": float(break_even_lmp),
        "cutoff_lmp": cutoff_lmp,
        "bx_evening_share_percent": float(evening.mean() * 100.0),
        "backup_evening_share_percent": (
            float(dropped_evening.mean() * 100.0) if not backup.empty else 0.0
        ),
    }
    return df, summary


def sensitivity_matrix(
    rows: Iterable[dict],
    assumptions: HybridAssumptions,
    *,
    citygate_values: Iterable[float],
    carbon_price_values: Iterable[float],
    transportation_per_mmbtu_in: float = 2.51,
    pppc_per_mmbtu_in: float = 0.66,
    emissions_ton_per_therm: float = 0.0053,
    gas_efficiency: float = 0.85,
) -> pd.DataFrame:
    """Return optimal hourly backup share across gas and carbon assumptions."""
    output = []
    normalized = _normalize_hours(rows).to_dict("records")
    for carbon_price in carbon_price_values:
        for citygate in citygate_values:
            gas = natural_gas_case(
                citygate_per_mmbtu_in=float(citygate),
                transportation_per_mmbtu_in=transportation_per_mmbtu_in,
                pppc_per_mmbtu_in=pppc_per_mmbtu_in,
                carbon_price_per_ton=float(carbon_price),
                emissions_ton_per_therm=emissions_ton_per_therm,
                gas_efficiency=gas_efficiency,
            )
            scenario = HybridAssumptions(
                **{
                    **assumptions.__dict__,
                    "backup_cost_per_mwh_th": gas["total_per_mwh_th"],
                }
            )
            _, summary = run_hybrid_dispatch(normalized, scenario, mode="economic")
            output.append({
                "citygate_per_mmbtu": float(citygate),
                "carbon_price_per_ton": float(carbon_price),
                "backup_share_percent": float(summary["backup_share_percent"]),
                "blended_cost_per_mwh_th": float(summary["blended_cost_per_mwh_th"]),
                "break_even_lmp": float(summary["break_even_lmp"]),
            })
    return pd.DataFrame(output)
