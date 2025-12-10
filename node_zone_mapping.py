"""
Node-to-Zone Mapping Module

Maps CAISO pricing nodes (PNODE_ID) to CAISO zones (NP15, SP15, ZP26, etc.)
using two mapping files:
  1. Zone mappings: Resource ID -> Matched Zone
  2. FNM mappings: PNODE_ID -> RES_ID

The join path is: NODE -> PNODE_ID -> RES_ID -> Resource ID -> Matched Zone

Usage:
    # Load with default files
    mapper = NodeZoneMapper()
    result = mapper.load_and_store_mappings()
    
    # Load with custom files
    mapper.load_mappings_from_csv(
        zone_file='path/to/zones.csv',
        fnm_file='path/to/fnm.csv'
    )
"""

import pandas as pd
import logging
from typing import Dict, Optional, List
from database import DatabaseManager

# Valid CAISO zones for analytics
VALID_ZONES = ['NP15', 'SP15', 'ZP26']

# Default file paths (can be overridden in function calls)
DEFAULT_ZONE_FILE = 'attached_assets/Zone_-_Node_Mappings_---_csv_1765396332817.csv'
DEFAULT_FNM_FILE = 'attached_assets/ATL_FNM_MAPPING_DATA_GRP_CISO_AS_24M3_DB127_v5_1765396332817.csv'


class NodeZoneMapper:
    """Handles loading and querying node-to-zone mappings"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.logger = logging.getLogger(__name__)
        self._mapping_cache: Dict[str, str] = {}
    
    def load_mappings_from_csv(
        self, 
        zone_file: str = None, 
        fnm_file: str = None
    ) -> pd.DataFrame:
        """
        Load and join the two mapping CSV files to create node-to-zone lookup.
        
        Args:
            zone_file: Path to zone mappings CSV (Resource ID -> Zone). 
                      Uses default if not provided.
            fnm_file: Path to FNM mappings CSV (PNODE_ID -> RES_ID).
                     Uses default if not provided.
        
        Returns:
            DataFrame with columns: pnode_id, resource_id, zone, local_area, generator_name
            Rows with no zone match will have zone=None (preserved for visibility)
        """
        # Use defaults if not provided
        zone_file = zone_file or DEFAULT_ZONE_FILE
        fnm_file = fnm_file or DEFAULT_FNM_FILE
        
        try:
            # Load zone mappings (Resource ID -> Zone)
            zone_df = pd.read_csv(zone_file, encoding='utf-8-sig')
            zone_df = zone_df[['Resource ID', 'Matched Zone', 'Local Area', 'Generator Name']].copy()
            zone_df.columns = ['resource_id', 'zone', 'local_area', 'generator_name']
            
            # Normalize zone values: "Not found" and empty strings become None
            zone_df['zone'] = zone_df['zone'].replace({'Not found': None, '': None})
            
            # Load FNM mappings (PNODE_ID -> RES_ID)
            fnm_df = pd.read_csv(fnm_file)
            fnm_df = fnm_df[['PNODE_ID', 'RES_ID']].copy()
            fnm_df.columns = ['pnode_id', 'resource_id']
            
            # Remove duplicates (same PNODE can appear multiple times with same RES_ID)
            fnm_df = fnm_df.drop_duplicates(subset=['pnode_id', 'resource_id'])
            
            # Join: PNODE_ID -> RES_ID -> Resource ID -> Zone
            # Using left join preserves all PNODEs even if no zone match
            merged = fnm_df.merge(zone_df, on='resource_id', how='left')
            
            # Remove duplicate pnode entries (take first match with a zone, else first overall)
            # Sort so rows with valid zones come first
            merged['has_zone'] = merged['zone'].notna()
            merged = merged.sort_values('has_zone', ascending=False)
            merged = merged.drop_duplicates(subset=['pnode_id'], keep='first')
            merged = merged.drop(columns=['has_zone'])
            
            mapped_count = merged['zone'].notna().sum()
            unmapped_count = merged['zone'].isna().sum()
            self.logger.info(f"Loaded {len(merged)} nodes: {mapped_count} mapped, {unmapped_count} unmapped")
            
            return merged
            
        except Exception as e:
            self.logger.error(f"Error loading mapping files: {str(e)}")
            raise
    
    def create_mapping_table(self) -> None:
        """Create the node_zone_mapping table in the database"""
        create_sql = """
        CREATE TABLE IF NOT EXISTS caiso.node_zone_mapping (
            id SERIAL PRIMARY KEY,
            pnode_id VARCHAR(100) NOT NULL UNIQUE,
            resource_id VARCHAR(100),
            zone VARCHAR(20),
            local_area VARCHAR(100),
            generator_name VARCHAR(200),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_node_zone_pnode ON caiso.node_zone_mapping(pnode_id);
        CREATE INDEX IF NOT EXISTS idx_node_zone_zone ON caiso.node_zone_mapping(zone);
        """
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(create_sql)
                    conn.commit()
            self.logger.info("Created node_zone_mapping table")
        except Exception as e:
            self.logger.error(f"Error creating mapping table: {str(e)}")
            raise
    
    def insert_mappings_to_db(self, df: pd.DataFrame) -> int:
        """
        Insert node-to-zone mappings into the database.
        
        Args:
            df: DataFrame with pnode_id, resource_id, zone, local_area, generator_name
            
        Returns:
            Number of records inserted
        """
        try:
            # Ensure table exists
            self.create_mapping_table()
            
            # Clear existing data
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("TRUNCATE TABLE caiso.node_zone_mapping RESTART IDENTITY")
                    conn.commit()
            
            # Prepare data for insertion
            df_clean = df[['pnode_id', 'resource_id', 'zone', 'local_area', 'generator_name']].copy()
            df_clean = df_clean.where(pd.notnull(df_clean), None)
            
            # Insert using pandas
            df_clean.to_sql(
                'node_zone_mapping',
                self.db.engine,
                schema='caiso',
                if_exists='append',
                index=False
            )
            
            self.logger.info(f"Inserted {len(df_clean)} node-zone mappings")
            return len(df_clean)
            
        except Exception as e:
            self.logger.error(f"Error inserting mappings: {str(e)}")
            raise
    
    def load_and_store_mappings(
        self, 
        zone_file: str = None, 
        fnm_file: str = None
    ) -> Dict[str, any]:
        """
        Complete workflow: Load CSVs and store in database.
        
        Args:
            zone_file: Optional path to zone mappings CSV
            fnm_file: Optional path to FNM mappings CSV
        
        Returns:
            Dict with 'total_nodes', 'mapped_nodes', 'unmapped_nodes', 'zones_found'
        """
        df = self.load_mappings_from_csv(zone_file, fnm_file)
        count = self.insert_mappings_to_db(df)
        
        # Count by zone (excludes None)
        zone_counts = df[df['zone'].notna()]['zone'].value_counts().to_dict()
        mapped_count = df['zone'].notna().sum()
        unmapped_count = df['zone'].isna().sum()
        
        return {
            'total_nodes': count,
            'mapped_nodes': int(mapped_count),
            'unmapped_nodes': int(unmapped_count),
            'zones_found': zone_counts
        }
    
    def get_zone_for_node(self, pnode_id: str) -> Optional[str]:
        """
        Look up the zone for a given pricing node.
        
        Args:
            pnode_id: The pricing node ID (matches NODE column in lmp_data)
            
        Returns:
            Zone string (NP15, SP15, ZP26) or None if not found
        """
        # Check cache first
        if pnode_id in self._mapping_cache:
            return self._mapping_cache[pnode_id]
        
        try:
            query = "SELECT zone FROM caiso.node_zone_mapping WHERE pnode_id = %s"
            result = self.db.execute_query(query, (pnode_id,))
            
            zone = result[0]['zone'] if result else None
            self._mapping_cache[pnode_id] = zone
            return zone
            
        except Exception as e:
            self.logger.error(f"Error looking up zone for {pnode_id}: {str(e)}")
            return None
    
    def get_nodes_for_zone(self, zone: str) -> List[str]:
        """
        Get all pricing nodes in a given zone.
        
        Args:
            zone: Zone string (NP15, SP15, ZP26)
            
        Returns:
            List of pnode_id strings
        """
        try:
            query = "SELECT pnode_id FROM caiso.node_zone_mapping WHERE zone = %s"
            result = self.db.execute_query(query, (zone,))
            return [row['pnode_id'] for row in result]
            
        except Exception as e:
            self.logger.error(f"Error getting nodes for zone {zone}: {str(e)}")
            return []
    
    def get_available_zones(self) -> List[str]:
        """Get list of zones that have mapped nodes"""
        try:
            query = """
                SELECT DISTINCT zone 
                FROM caiso.node_zone_mapping 
                WHERE zone IS NOT NULL 
                ORDER BY zone
            """
            result = self.db.execute_query(query)
            return [row['zone'] for row in result]
            
        except Exception as e:
            self.logger.error(f"Error getting available zones: {str(e)}")
            return VALID_ZONES
    
    def get_mapping_stats(self) -> Dict[str, any]:
        """Get statistics about the current mappings"""
        try:
            query = """
                SELECT 
                    COUNT(*) as total_mappings,
                    COUNT(DISTINCT zone) as unique_zones,
                    COUNT(CASE WHEN zone IS NULL THEN 1 END) as unmapped_nodes
                FROM caiso.node_zone_mapping
            """
            result = self.db.execute_query(query)
            
            zone_query = """
                SELECT zone, COUNT(*) as node_count 
                FROM caiso.node_zone_mapping 
                WHERE zone IS NOT NULL 
                GROUP BY zone 
                ORDER BY zone
            """
            zone_result = self.db.execute_query(zone_query)
            
            return {
                'total_mappings': result[0]['total_mappings'] if result else 0,
                'unique_zones': result[0]['unique_zones'] if result else 0,
                'unmapped_nodes': result[0]['unmapped_nodes'] if result else 0,
                'nodes_per_zone': {row['zone']: row['node_count'] for row in zone_result}
            }
            
        except Exception as e:
            self.logger.error(f"Error getting mapping stats: {str(e)}")
            return {}


# Convenience function for quick loading
def load_zone_mappings(zone_file: str = None, fnm_file: str = None) -> Dict[str, any]:
    """
    Load zone mappings from CSV files into database.
    
    Args:
        zone_file: Optional path to zone mappings CSV
        fnm_file: Optional path to FNM mappings CSV
        
    Returns:
        Dict with total_nodes, mapped_nodes, unmapped_nodes, zones_found
    """
    mapper = NodeZoneMapper()
    return mapper.load_and_store_mappings(zone_file, fnm_file)


if __name__ == "__main__":
    # Test the module
    logging.basicConfig(level=logging.INFO)
    
    print("Loading node-zone mappings...")
    result = load_zone_mappings()
    print(f"Result: {result}")
    
    mapper = NodeZoneMapper()
    stats = mapper.get_mapping_stats()
    print(f"Stats: {stats}")
