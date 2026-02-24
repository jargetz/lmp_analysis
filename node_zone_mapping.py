"""
Node-to-Zone Mapping Module

Maps CAISO pricing nodes (PNODE_ID) to CAISO zones (NP15, SP15, ZP26)
using CAISO AS (Ancillary Services) Region Map files.

Zone inference logic:
  - Node in both AS_NP15 and AS_NP26 → NP15
  - Node in both AS_SP15 and AS_NP26 → ZP26
  - Node in AS_SP15 only (no AS_NP26) → SP15
  - Node in AS_NP15_EXP only → NP15 (export/intertie nodes)

Also supports APNode mapping (which component nodes make up each
aggregated pricing node like TH_NP15_GEN-APND).
"""

import csv
import logging
from typing import Dict, Optional, List
from collections import defaultdict

VALID_ZONES = ['NP15', 'SP15', 'ZP26']

AS_REGION_FILES = {
    'attached_assets/20260224_20260225_ATL_AS_REGION_MAP_N_20260224_11_25_31_v1_1771961685690.csv': 'AS_NP26',
    'attached_assets/20260224_20260225_ATL_AS_REGION_MAP_N_20260224_11_23_07_v1_1771961685690.csv': 'AS_SP15',
    'attached_assets/20260224_20260225_ATL_AS_REGION_MAP_N_20260224_11_21_08_v1_1771961685690.csv': 'AS_NP15',
    'attached_assets/20260224_20260225_ATL_AS_REGION_MAP_N_20260224_11_19_56_v1_1771961685690.csv': 'AS_NP15_EXP',
}

APNODE_FILE = 'attached_assets/20260224_20260225_ATL_PNODE_MAP_N_20260224_11_18_05_v1_1771961685690.csv'


def infer_zone_from_regions(zone_set):
    if {"AS_NP15", "AS_NP26"}.issubset(zone_set):
        return "NP15"
    elif {"AS_SP15", "AS_NP26"}.issubset(zone_set):
        return "ZP26"
    elif {"AS_SP15", "AS_SP26"}.issubset(zone_set):
        return "SP15"
    elif "AS_SP15" in zone_set and "AS_NP26" not in zone_set:
        return "SP15"
    elif "AS_NP15_EXP" in zone_set:
        return "NP15"
    return None


class NodeZoneMapper:
    def __init__(self):
        self._md = None
        self.logger = logging.getLogger(__name__)
        self._mapping_cache: Dict[str, str] = {}

    def _get_md(self):
        if self._md is None:
            from motherduck_client import get_motherduck_client
            self._md = get_motherduck_client(force_new=True)
        return self._md

    def load_mappings_from_as_regions(self) -> List[tuple]:
        pnode_regions = defaultdict(set)
        for fpath in AS_REGION_FILES:
            with open(fpath) as f:
                for row in csv.DictReader(f):
                    pnode_regions[row['PNODE_ID']].add(row['AS_REGION_ID'])

        mappings = []
        for pnode, zone_set in sorted(pnode_regions.items()):
            zone = infer_zone_from_regions(zone_set)
            mappings.append((pnode, zone))

        self.logger.info(f"Loaded {len(mappings)} node-zone mappings from AS region files")
        return mappings

    def load_and_store_mappings(self) -> Dict[str, any]:
        mappings = self.load_mappings_from_as_regions()

        md = self._get_md()
        md.conn.execute("DELETE FROM node_zone_mapping")

        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='')
        writer = csv.writer(tmp)
        writer.writerow(['pnode_id', 'zone'])
        for row in mappings:
            writer.writerow(row)
        tmp.close()

        md.conn.execute(f"""
            INSERT INTO node_zone_mapping (pnode_id, zone)
            SELECT pnode_id, zone FROM read_csv('{tmp.name}', header=true)
        """)
        os.unlink(tmp.name)

        zone_counts = defaultdict(int)
        for _, zone in mappings:
            if zone:
                zone_counts[zone] += 1

        return {
            'total_nodes': len(mappings),
            'mapped_nodes': sum(1 for _, z in mappings if z),
            'unmapped_nodes': sum(1 for _, z in mappings if not z),
            'zones_found': dict(zone_counts)
        }

    def load_apnode_mappings(self) -> int:
        md = self._get_md()
        md.conn.execute("""
            CREATE TABLE IF NOT EXISTS node_apnode_mapping (
                apnode_id VARCHAR,
                pnode_id VARCHAR
            )
        """)
        md.conn.execute("DELETE FROM node_apnode_mapping")
        md.conn.execute(f"""
            INSERT INTO node_apnode_mapping (apnode_id, pnode_id)
            SELECT APNODE_ID, PNODE_ID FROM read_csv('{APNODE_FILE}', header=true)
        """)
        result = md.conn.execute("SELECT COUNT(*) as cnt FROM node_apnode_mapping").fetchone()
        count = result[0]
        self.logger.info(f"Loaded {count} APNode mappings")
        return count

    def get_zone_for_node(self, pnode_id: str) -> Optional[str]:
        if pnode_id in self._mapping_cache:
            return self._mapping_cache[pnode_id]

        try:
            md = self._get_md()
            query = "SELECT zone FROM node_zone_mapping WHERE pnode_id = $1"
            result = md.execute_query(query, (pnode_id,))
            zone = result[0]['zone'] if result else None
            self._mapping_cache[pnode_id] = zone
            return zone
        except Exception as e:
            self.logger.error(f"Error looking up zone for {pnode_id}: {str(e)}")
            return None

    def get_nodes_for_zone(self, zone: str) -> List[str]:
        try:
            md = self._get_md()
            query = "SELECT pnode_id FROM node_zone_mapping WHERE zone = $1"
            result = md.execute_query(query, (zone,))
            return [row['pnode_id'] for row in result]
        except Exception as e:
            self.logger.error(f"Error getting nodes for zone {zone}: {str(e)}")
            return []

    def get_available_zones(self) -> List[str]:
        try:
            md = self._get_md()
            query = """
                SELECT DISTINCT zone 
                FROM node_zone_mapping 
                WHERE zone IS NOT NULL 
                ORDER BY zone
            """
            result = md.execute_query(query)
            return [row['zone'] for row in result]
        except Exception as e:
            self.logger.error(f"Error getting available zones: {str(e)}")
            return VALID_ZONES

    def get_mapping_stats(self) -> Dict[str, any]:
        try:
            md = self._get_md()
            query = """
                SELECT 
                    COUNT(*) as total_mappings,
                    COUNT(DISTINCT zone) as unique_zones,
                    COUNT(CASE WHEN zone IS NULL THEN 1 END) as unmapped_nodes
                FROM node_zone_mapping
            """
            result = md.execute_query(query)

            zone_query = """
                SELECT zone, COUNT(*) as node_count 
                FROM node_zone_mapping 
                WHERE zone IS NOT NULL 
                GROUP BY zone 
                ORDER BY zone
            """
            zone_result = md.execute_query(zone_query)

            return {
                'total_mappings': result[0]['total_mappings'] if result else 0,
                'unique_zones': result[0]['unique_zones'] if result else 0,
                'unmapped_nodes': result[0]['unmapped_nodes'] if result else 0,
                'nodes_per_zone': {row['zone']: row['node_count'] for row in zone_result}
            }
        except Exception as e:
            self.logger.error(f"Error getting mapping stats: {str(e)}")
            return {}


def load_zone_mappings() -> Dict[str, any]:
    mapper = NodeZoneMapper()
    return mapper.load_and_store_mappings()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Loading node-zone mappings from AS region files...")
    result = load_zone_mappings()
    print(f"Result: {result}")

    mapper = NodeZoneMapper()
    stats = mapper.get_mapping_stats()
    print(f"Stats: {stats}")

    print("\nLoading APNode mappings...")
    count = mapper.load_apnode_mappings()
    print(f"Loaded {count} APNode mappings")
