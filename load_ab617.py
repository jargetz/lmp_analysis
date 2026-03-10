"""
Load AB 617 Community Air Protection Program communities from CARB's ArcGIS API
into MotherDuck as caiso_lmp.ab617_communities.

Source: CARB ArcGIS FeatureServer (consistently nominated communities, 2023 list)
"""
import os
import json
import duckdb
import urllib.request

CARB_URL = (
    "https://gis.carb.arb.ca.gov/hosting/rest/services/Hosted/"
    "AB_617_Communities/FeatureServer/1/query"
    "?where=1%3D1&outFields=*&f=geojson"
)

def fetch_ab617_communities():
    """Fetch AB 617 community features from CARB ArcGIS endpoint."""
    print(f"Fetching AB 617 data from CARB ArcGIS API...")
    req = urllib.request.Request(CARB_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    features = data.get("features", [])
    print(f"  Raw features returned: {len(features)}")
    return features

def parse_communities(features):
    """Extract community records with valid coordinates.
    Note: CARB's API has field names swapped:
      'centroid_longtitude' actually contains latitude values (~32-42)
      'centroid_latitude'   actually contains longitude values (~-114 to -125)
    """
    records = []
    for feat in features:
        props = feat.get("properties") or {}
        # Swap: field named 'longtitude' holds lat, field named 'latitude' holds lon
        lat = props.get("centroid_longtitude")
        lon = props.get("centroid_latitude")
        name = props.get("ssl_name") or props.get("Name") or props.get("name")
        if lat is None or lon is None or name is None:
            continue
        try:
            lat = float(lat)
            lon = float(lon)
        except (TypeError, ValueError):
            continue
        if not (-125 <= lon <= -114 and 32 <= lat <= 42):
            continue
        records.append({
            "community_name": str(name).strip(),
            "lat": lat,
            "lon": lon,
        })
    return records

def load_to_motherduck(records):
    """Create/replace ab617_communities table in MotherDuck."""
    token = os.getenv("MOTHERDUCK_TOKEN")
    if not token:
        raise RuntimeError("MOTHERDUCK_TOKEN not set")
    conn = duckdb.connect(f"md:?motherduck_token={token}")
    conn.execute("SET enable_progress_bar = false")
    conn.execute("USE caiso_lmp")

    conn.execute("DROP TABLE IF EXISTS ab617_communities")
    conn.execute("""
        CREATE TABLE ab617_communities (
            community_name VARCHAR,
            lat DOUBLE,
            lon DOUBLE
        )
    """)

    for r in records:
        conn.execute(
            "INSERT INTO ab617_communities VALUES (?, ?, ?)",
            [r["community_name"], r["lat"], r["lon"]]
        )

    count = conn.execute("SELECT COUNT(*) FROM ab617_communities").fetchone()[0]
    print(f"  Loaded {count} AB 617 communities into MotherDuck")

    sample = conn.execute(
        "SELECT community_name, ROUND(lat,4) AS lat, ROUND(lon,4) AS lon "
        "FROM ab617_communities ORDER BY community_name LIMIT 10"
    ).fetchdf()
    print(sample.to_string(index=False))
    conn.close()
    return count

if __name__ == "__main__":
    features = fetch_ab617_communities()
    records = parse_communities(features)
    print(f"  Valid CA communities parsed: {len(records)}")
    if not records:
        print("ERROR: No valid records found — check API response")
        raise SystemExit(1)
    n = load_to_motherduck(records)
    print(f"Done. {n} communities loaded.")
