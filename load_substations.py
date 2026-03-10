"""
Load CA substation data into MotherDuck and compute node→nearest-substation mapping.

Tables created:
  ca_substations          — 3,261 CA substations with location, owner, voltage class, status
  node_substation_mapping — each pnode mapped to its nearest substation by geography

Usage:
  python3 load_substations.py
"""

import os
import math
import numpy as np
import pandas as pd
import duckdb

CSV_PATH = 'attached_assets/CA_Substation_Coordinates_1773167584973.csv'


def normalize_highest_kv(val: str) -> str:
    if isinstance(val, str):
        val = val.strip()
        if val == '33kV to 92Kv':
            val = '33kV to 92kV'
    return val


def main():
    token = os.environ['MOTHERDUCK_TOKEN']
    conn = duckdb.connect(f'md:caiso_lmp?motherduck_token={token}')

    print('Reading substation CSV…')
    df = pd.read_csv(CSV_PATH, dtype=str)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()

    keep = {
        'X': 'lon',
        'Y': 'lat',
        'Substation_ID': 'substation_id',
        'Substation_Name': 'substation_name',
        'Alias': 'alias',
        'Status': 'status',
        'Owner': 'owner',
        'kV_12_TO_32': 'kv_12_to_32',
        'kV_33_TO_92': 'kv_33_to_92',
        'kV_110_TO_161': 'kv_110_to_161',
        'kV_220_To_287': 'kv_220_to_287',
        'kV_345_To_500': 'kv_345_to_500',
        'kV_500_DC': 'kv_500_dc',
        'Highest_kV': 'highest_kv',
        'Postal_City': 'city',
        'County': 'county',
    }
    df = df.rename(columns=keep)[list(keep.values())]

    df['highest_kv'] = df['highest_kv'].apply(normalize_highest_kv)
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')

    print(f'  {len(df)} rows loaded')
    print(f"  Status values: {df['status'].value_counts().to_dict()}")
    print(f"  Highest_kV values: {df['highest_kv'].value_counts().to_dict()}")

    print('\nLoading ca_substations into MotherDuck…')
    conn.execute('DROP TABLE IF EXISTS caiso_lmp.ca_substations')
    conn.execute("""
        CREATE TABLE caiso_lmp.ca_substations AS
        SELECT * FROM df
    """)
    n = conn.execute('SELECT COUNT(*) FROM caiso_lmp.ca_substations').fetchone()[0]
    print(f'  Loaded {n} rows into ca_substations')

    print('\nLoading pnode_coordinates from MotherDuck…')
    pnodes = conn.execute(
        'SELECT pnode_id, lat, lon, node_type, area FROM caiso_lmp.pnode_coordinates'
    ).fetchdf()
    print(f'  {len(pnodes)} pnodes loaded')

    sub_op = df.dropna(subset=['lat', 'lon']).copy().reset_index(drop=True)
    sub_lats = sub_op['lat'].values
    sub_lons = sub_op['lon'].values
    print(f'  {len(sub_op)} substations have coordinates')

    print('\nComputing nearest substation for each pnode (vectorized)…')
    records = []
    batch = 500
    n_pnodes = len(pnodes)
    for start in range(0, n_pnodes, batch):
        chunk = pnodes.iloc[start:start + batch]
        for _, row in chunk.iterrows():
            plat = float(row['lat'])
            plon = float(row['lon'])
            cos_lat = math.cos(math.radians(plat))
            dlat = sub_lats - plat
            dlon = (sub_lons - plon) * cos_lat
            dists = np.sqrt(dlat ** 2 + dlon ** 2) * 111.0
            idx = int(dists.argmin())
            sub = sub_op.iloc[idx]
            records.append({
                'pnode_id': str(row['pnode_id']),
                'node_type': str(row['node_type']),
                'substation_id': str(sub['substation_id']),
                'substation_name': str(sub['substation_name']),
                'owner': str(sub['owner']),
                'status': str(sub['status']),
                'highest_kv': str(sub['highest_kv']) if pd.notna(sub['highest_kv']) else None,
                'dist_km': float(dists[idx]),
            })
        if (start + batch) % 2000 == 0 or start + batch >= n_pnodes:
            print(f'  {min(start + batch, n_pnodes):,}/{n_pnodes:,} pnodes processed…')

    mapping_df = pd.DataFrame(records)

    print('\nSanity check — distance distribution:')
    for label, lo, hi in [
        ('≤1 km', 0, 1), ('1–5 km', 1, 5), ('5–10 km', 5, 10),
        ('10–50 km', 10, 50), ('>50 km', 50, 1e9)
    ]:
        n = ((mapping_df['dist_km'] > lo) & (mapping_df['dist_km'] <= hi)).sum() if lo > 0 else (mapping_df['dist_km'] <= hi).sum()
        pct = 100 * n / len(mapping_df)
        print(f'  {label}: {n:,} ({pct:.1f}%)')

    print('\nBy node_type:')
    for nt, grp in mapping_df.groupby('node_type'):
        med = grp['dist_km'].median()
        p90 = grp['dist_km'].quantile(0.9)
        print(f'  {nt}: {len(grp):,} nodes, median dist={med:.1f}km, p90={p90:.1f}km')

    non_op = mapping_df[mapping_df['status'] != 'Operational']
    if not non_op.empty:
        print(f'\n  ⚠ {len(non_op)} pnodes matched to NON-OPERATIONAL substations:')
        for _, r in non_op.head(10).iterrows():
            print(f"    {r['pnode_id']} → {r['substation_name']} ({r['status']}, {r['dist_km']:.1f}km)")
        if len(non_op) > 10:
            print(f'    … and {len(non_op) - 10} more')

    print('\nLoading node_substation_mapping into MotherDuck…')
    conn.execute('DROP TABLE IF EXISTS caiso_lmp.node_substation_mapping')
    conn.execute("""
        CREATE TABLE caiso_lmp.node_substation_mapping AS
        SELECT * FROM mapping_df
    """)
    n = conn.execute('SELECT COUNT(*) FROM caiso_lmp.node_substation_mapping').fetchone()[0]
    print(f'  Loaded {n} rows into node_substation_mapping')

    conn.close()
    print('\nDone.')


if __name__ == '__main__':
    main()
