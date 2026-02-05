"""
Database connector for PostgreSQL graph storage.

This module handles all PostgreSQL operations for storing and retrieving
the knowledge graph as edges in a relational table.
"""

import os
import sys

# Fix encoding issues on Windows before importing psycopg2
if sys.platform == 'win32':
    # Set locale to C (ASCII) to avoid UTF-8 decoding errors with system paths
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'C')
    except:
        pass
    # Set environment variables for PostgreSQL client
    os.environ['LC_ALL'] = 'C'
    os.environ['LANG'] = 'C'
    os.environ['PGCLIENTENCODING'] = 'UTF8'

import psycopg2
from psycopg2.extras import execute_batch
import networkx as nx
from typing import Optional, List, Tuple
import logging


class GraphDatabaseConnector:
    """
    Connector class for storing and retrieving graphs from PostgreSQL.

    The graph is stored as a simple edge table:
    - source (TEXT): Source node ID
    - predicate (TEXT): Edge type/predicate
    - target (TEXT): Target node ID
    - weight (FLOAT): Edge weight (default 1.0)
    """

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        graph_id: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the database connector.

        Args:
            host: PostgreSQL host
            port: PostgreSQL port
            database: Database name
            user: Username
            password: Password
            graph_id: Optional graph ID (timestamp) for versioned tables
            logger: Optional logger instance
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.graph_id = graph_id
        self.logger = logger or logging.getLogger(__name__)

        self.conn = None
        self.cursor = None

        # Table names (with optional graph_id suffix for versioning)
        if graph_id:
            self.graph_table = f"graph_{graph_id}"
            self.nodes_table = f"nodes_{graph_id}"
        else:
            self.graph_table = "graph"
            self.nodes_table = "nodes"

    def connect(self):
        """Establish connection to PostgreSQL database."""
        try:
            # Clean environment variables that might contain non-UTF8 characters
            env_backup = {}
            problematic_vars = ['APPDATA', 'LOCALAPPDATA', 'TEMP', 'TMP', 'PGPASSFILE', 'PGSYSCONFDIR']

            for var in problematic_vars:
                if var in os.environ:
                    env_backup[var] = os.environ[var]
                    del os.environ[var]

            try:
                # Set PostgreSQL encoding
                os.environ['PGCLIENTENCODING'] = 'UTF8'

                # Try connection with explicit parameters (no DSN parsing)
                self.conn = psycopg2.connect(
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    user=self.user,
                    password=self.password,
                    options='-c client_encoding=UTF8'
                )
            finally:
                # Restore environment variables
                for var, value in env_backup.items():
                    os.environ[var] = value

            self.cursor = self.conn.cursor()

            # Set connection encoding explicitly after connection
            self.conn.set_client_encoding('UTF8')

            self.logger.info(f"Connected to PostgreSQL database: {self.database}")
        except Exception as e:
            self.logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    def disconnect(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        self.logger.info("Disconnected from PostgreSQL")

    def create_tables(self):
        """Create the graph table if it doesn't exist."""
        create_graph_table = f"""
        CREATE TABLE IF NOT EXISTS {self.graph_table} (
            id SERIAL PRIMARY KEY,
            source TEXT NOT NULL,
            predicate TEXT NOT NULL,
            target TEXT NOT NULL,
            weight FLOAT DEFAULT 1.0,
            attributes JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        # Create indexes for better query performance
        create_indexes = f"""
        CREATE INDEX IF NOT EXISTS idx_{self.graph_table}_source ON {self.graph_table}(source);
        CREATE INDEX IF NOT EXISTS idx_{self.graph_table}_target ON {self.graph_table}(target);
        CREATE INDEX IF NOT EXISTS idx_{self.graph_table}_predicate ON {self.graph_table}(predicate);
        CREATE INDEX IF NOT EXISTS idx_{self.graph_table}_source_target ON {self.graph_table}(source, target);
        """

        # Create node attributes table for storing node metadata
        create_node_table = f"""
        CREATE TABLE IF NOT EXISTS {self.nodes_table} (
            node_id TEXT PRIMARY KEY,
            node_type TEXT,
            attributes JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        create_node_index = f"""
        CREATE INDEX IF NOT EXISTS idx_{self.nodes_table}_type ON {self.nodes_table}(node_type);
        """

        # Create metadata table (only once, not versioned)
        create_metadata_table = """
        CREATE TABLE IF NOT EXISTS graph_metadata (
            graph_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            num_nodes INTEGER,
            num_edges INTEGER,
            metadata JSONB
        );
        """

        try:
            self.cursor.execute(create_graph_table)
            self.cursor.execute(create_indexes)
            self.cursor.execute(create_node_table)
            self.cursor.execute(create_node_index)
            self.cursor.execute(create_metadata_table)

            # Migrate existing tables: add attributes column if missing
            self._migrate_add_attributes_column()

            self.conn.commit()
            self.logger.info(f"Graph tables created successfully: {self.graph_table}, {self.nodes_table}")
        except Exception as e:
            self.conn.rollback()
            self.logger.error(f"Failed to create tables: {e}")
            raise

    def _migrate_add_attributes_column(self):
        """Add attributes column to edge table if it doesn't exist (migration)."""
        try:
            # Check if attributes column exists
            self.cursor.execute(f"""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = '{self.graph_table}'
                AND column_name = 'attributes'
            """)

            if not self.cursor.fetchone():
                self.logger.info(f"Migrating {self.graph_table}: adding 'attributes' column...")
                self.cursor.execute(f"""
                    ALTER TABLE {self.graph_table}
                    ADD COLUMN attributes JSONB
                """)
                self.logger.info(f"Migration complete: 'attributes' column added to {self.graph_table}")
        except Exception as e:
            self.logger.warning(f"Migration warning: {e}")

    def clear_graph(self):
        """Clear all data from graph and nodes tables."""
        try:
            self.cursor.execute(f"DELETE FROM {self.graph_table}")
            self.cursor.execute(f"DELETE FROM {self.nodes_table}")
            self.conn.commit()
            self.logger.info(f"Graph tables cleared: {self.graph_table}, {self.nodes_table}")
        except Exception as e:
            self.conn.rollback()
            self.logger.error(f"Failed to clear tables: {e}")
            raise

    def save_graph(self, graph: nx.Graph, metadata: dict = None):
        """
        Save a NetworkX graph to PostgreSQL.

        Args:
            graph: NetworkX graph to save
            metadata: Optional metadata dictionary to store
        """
        self.logger.info("Saving graph to PostgreSQL...")

        # Prepare node data
        node_data = []
        for node, attrs in graph.nodes(data=True):
            # Convert attributes to JSON-compatible format
            clean_attrs = {}
            for key, value in attrs.items():
                # Convert numpy types, lists, etc. to JSON-compatible types
                if hasattr(value, 'tolist'):
                    clean_attrs[key] = value.tolist()
                elif isinstance(value, (list, dict, str, int, float, bool, type(None))):
                    clean_attrs[key] = value
                else:
                    clean_attrs[key] = str(value)

            node_type = attrs.get('type', 'unknown')
            node_data.append((node, node_type, psycopg2.extras.Json(clean_attrs)))

        # Prepare edge data
        edge_data = []
        for source, target, attrs in graph.edges(data=True):
            edge_type = attrs.get('edge_type', 'UNKNOWN')
            weight = attrs.get('weight', 1.0)

            # Convert all edge attributes to JSON-compatible format (similar to nodes)
            clean_attrs = {}
            for key, value in attrs.items():
                if key in ('edge_type', 'weight'):
                    continue  # Already stored in dedicated columns
                # Convert numpy types, lists, etc. to JSON-compatible types
                if hasattr(value, 'tolist'):
                    clean_attrs[key] = value.tolist()
                elif isinstance(value, (list, dict, str, int, float, bool, type(None))):
                    clean_attrs[key] = value
                else:
                    clean_attrs[key] = str(value)

            edge_data.append((source, edge_type, target, float(weight), psycopg2.extras.Json(clean_attrs)))

        try:
            # Insert nodes in batches
            self.logger.info(f"Inserting {len(node_data)} nodes into {self.nodes_table}...")
            insert_node_query = f"""
            INSERT INTO {self.nodes_table} (node_id, node_type, attributes)
            VALUES (%s, %s, %s)
            ON CONFLICT (node_id) DO UPDATE SET
                node_type = EXCLUDED.node_type,
                attributes = EXCLUDED.attributes
            """
            execute_batch(self.cursor, insert_node_query, node_data, page_size=1000)

            # Insert edges in batches
            self.logger.info(f"Inserting {len(edge_data)} edges into {self.graph_table}...")
            insert_edge_query = f"""
            INSERT INTO {self.graph_table} (source, predicate, target, weight, attributes)
            VALUES (%s, %s, %s, %s, %s)
            """
            execute_batch(self.cursor, insert_edge_query, edge_data, page_size=1000)

            # Insert metadata if graph_id is provided
            if self.graph_id and metadata:
                self.logger.info(f"Inserting metadata for graph {self.graph_id}...")
                insert_metadata_query = """
                INSERT INTO graph_metadata (graph_id, num_nodes, num_edges, metadata)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (graph_id) DO UPDATE SET
                    num_nodes = EXCLUDED.num_nodes,
                    num_edges = EXCLUDED.num_edges,
                    metadata = EXCLUDED.metadata
                """
                self.cursor.execute(
                    insert_metadata_query,
                    (self.graph_id, len(node_data), len(edge_data), psycopg2.extras.Json(metadata))
                )

            self.conn.commit()
            self.logger.info(f"Graph saved successfully: {len(node_data)} nodes, {len(edge_data)} edges")

        except Exception as e:
            self.conn.rollback()
            self.logger.error(f"Failed to save graph: {e}")
            raise

    def load_graph(self) -> nx.Graph:
        """
        Load the complete graph from PostgreSQL.

        Returns:
            NetworkX graph reconstructed from database
        """
        self.logger.info(f"Loading graph from PostgreSQL (tables: {self.nodes_table}, {self.graph_table})...")

        graph = nx.Graph()

        try:
            # Load nodes
            self.cursor.execute(f"SELECT node_id, node_type, attributes FROM {self.nodes_table}")
            nodes = self.cursor.fetchall()

            self.logger.info(f"Loading {len(nodes)} nodes...")
            for node_id, node_type, attributes in nodes:
                graph.add_node(node_id, **attributes)

            # Load edges
            self.cursor.execute(f"SELECT source, predicate, target, weight, attributes FROM {self.graph_table}")
            edges = self.cursor.fetchall()

            self.logger.info(f"Loading {len(edges)} edges...")
            for source, predicate, target, weight, attributes in edges:
                # Merge edge attributes from JSONB with basic attributes
                edge_attrs = {'edge_type': predicate, 'weight': weight}
                if attributes:
                    edge_attrs.update(attributes)
                graph.add_edge(source, target, **edge_attrs)

            self.logger.info(f"Graph loaded successfully: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            return graph

        except Exception as e:
            self.logger.error(f"Failed to load graph: {e}")
            raise

    def get_graph_statistics(self) -> dict:
        """
        Get statistics about the stored graph.

        Returns:
            Dictionary with graph statistics
        """
        try:
            stats = {}

            # Node counts
            self.cursor.execute(f"SELECT COUNT(*) FROM {self.nodes_table}")
            stats['total_nodes'] = self.cursor.fetchone()[0]

            self.cursor.execute(f"SELECT node_type, COUNT(*) FROM {self.nodes_table} GROUP BY node_type")
            stats['nodes_by_type'] = dict(self.cursor.fetchall())

            # Edge counts
            self.cursor.execute(f"SELECT COUNT(*) FROM {self.graph_table}")
            stats['total_edges'] = self.cursor.fetchone()[0]

            self.cursor.execute(f"SELECT predicate, COUNT(*) FROM {self.graph_table} GROUP BY predicate")
            stats['edges_by_type'] = dict(self.cursor.fetchall())

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            raise

    @staticmethod
    def list_available_graphs(
        host: str = "localhost",
        port: int = 5432,
        database: str = "knowledge_graph",
        user: str = "postgres",
        password: str = "postgres"
    ) -> List[dict]:
        """
        List all available graphs in the database.

        Returns:
            List of dictionaries with graph metadata
        """
        conn = None
        try:
            conn = psycopg2.connect(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password
            )
            cursor = conn.cursor()

            cursor.execute("""
                SELECT graph_id, created_at, num_nodes, num_edges, metadata
                FROM graph_metadata
                ORDER BY created_at DESC
            """)

            results = []
            for graph_id, created_at, num_nodes, num_edges, metadata in cursor.fetchall():
                results.append({
                    'graph_id': graph_id,
                    'created_at': created_at,
                    'num_nodes': num_nodes,
                    'num_edges': num_edges,
                    'metadata': metadata
                })

            cursor.close()
            return results

        except Exception as e:
            raise
        finally:
            if conn:
                conn.close()

    def query_neighbors(
        self,
        node_id: str,
        edge_type: Optional[str] = None,
        max_neighbors: int = 100
    ) -> List[Tuple[str, str, float]]:
        """
        Query neighbors of a specific node.

        Args:
            node_id: Node to query
            edge_type: Optional edge type filter
            max_neighbors: Maximum number of neighbors to return

        Returns:
            List of (neighbor_id, edge_type, weight) tuples
        """
        try:
            if edge_type:
                query = f"""
                SELECT target, predicate, weight FROM {self.graph_table}
                WHERE source = %s AND predicate = %s
                UNION
                SELECT source, predicate, weight FROM {self.graph_table}
                WHERE target = %s AND predicate = %s
                LIMIT %s
                """
                self.cursor.execute(query, (node_id, edge_type, node_id, edge_type, max_neighbors))
            else:
                query = f"""
                SELECT target, predicate, weight FROM {self.graph_table}
                WHERE source = %s
                UNION
                SELECT source, predicate, weight FROM {self.graph_table}
                WHERE target = %s
                LIMIT %s
                """
                self.cursor.execute(query, (node_id, node_id, max_neighbors))

            return self.cursor.fetchall()

        except Exception as e:
            self.logger.error(f"Failed to query neighbors: {e}")
            raise
