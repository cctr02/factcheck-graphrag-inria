"""
Database connector for PostgreSQL graph storage.

This module handles all PostgreSQL operations for storing and retrieving
the knowledge graph as edges in a relational table.
"""

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
        host: str = "localhost",
        port: int = 5432,
        database: str = "knowledge_graph",
        user: str = "postgres",
        password: str = "postgres",
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
            logger: Optional logger instance
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.logger = logger or logging.getLogger(__name__)

        self.conn = None
        self.cursor = None

    def connect(self):
        """Establish connection to PostgreSQL database."""
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            self.cursor = self.conn.cursor()
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
        create_graph_table = """
        CREATE TABLE IF NOT EXISTS graph (
            id SERIAL PRIMARY KEY,
            source TEXT NOT NULL,
            predicate TEXT NOT NULL,
            target TEXT NOT NULL,
            weight FLOAT DEFAULT 1.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        # Create indexes for better query performance
        create_indexes = """
        CREATE INDEX IF NOT EXISTS idx_graph_source ON graph(source);
        CREATE INDEX IF NOT EXISTS idx_graph_target ON graph(target);
        CREATE INDEX IF NOT EXISTS idx_graph_predicate ON graph(predicate);
        CREATE INDEX IF NOT EXISTS idx_graph_source_target ON graph(source, target);
        """

        # Create node attributes table for storing node metadata
        create_node_table = """
        CREATE TABLE IF NOT EXISTS nodes (
            node_id TEXT PRIMARY KEY,
            node_type TEXT,
            attributes JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        create_node_index = """
        CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
        """

        try:
            self.cursor.execute(create_graph_table)
            self.cursor.execute(create_indexes)
            self.cursor.execute(create_node_table)
            self.cursor.execute(create_node_index)
            self.conn.commit()
            self.logger.info("Graph tables created successfully")
        except Exception as e:
            self.conn.rollback()
            self.logger.error(f"Failed to create tables: {e}")
            raise

    def clear_graph(self):
        """Clear all data from graph and nodes tables."""
        try:
            self.cursor.execute("DELETE FROM graph")
            self.cursor.execute("DELETE FROM nodes")
            self.conn.commit()
            self.logger.info("Graph tables cleared")
        except Exception as e:
            self.conn.rollback()
            self.logger.error(f"Failed to clear tables: {e}")
            raise

    def save_graph(self, graph: nx.Graph):
        """
        Save a NetworkX graph to PostgreSQL.

        Args:
            graph: NetworkX graph to save
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
            edge_data.append((source, edge_type, target, float(weight)))

        try:
            # Insert nodes in batches
            self.logger.info(f"Inserting {len(node_data)} nodes...")
            insert_node_query = """
            INSERT INTO nodes (node_id, node_type, attributes)
            VALUES (%s, %s, %s)
            ON CONFLICT (node_id) DO UPDATE SET
                node_type = EXCLUDED.node_type,
                attributes = EXCLUDED.attributes
            """
            execute_batch(self.cursor, insert_node_query, node_data, page_size=1000)

            # Insert edges in batches
            self.logger.info(f"Inserting {len(edge_data)} edges...")
            insert_edge_query = """
            INSERT INTO graph (source, predicate, target, weight)
            VALUES (%s, %s, %s, %s)
            """
            execute_batch(self.cursor, insert_edge_query, edge_data, page_size=1000)

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
        self.logger.info("Loading graph from PostgreSQL...")

        graph = nx.Graph()

        try:
            # Load nodes
            self.cursor.execute("SELECT node_id, node_type, attributes FROM nodes")
            nodes = self.cursor.fetchall()

            self.logger.info(f"Loading {len(nodes)} nodes...")
            for node_id, node_type, attributes in nodes:
                graph.add_node(node_id, **attributes)

            # Load edges
            self.cursor.execute("SELECT source, predicate, target, weight FROM graph")
            edges = self.cursor.fetchall()

            self.logger.info(f"Loading {len(edges)} edges...")
            for source, predicate, target, weight in edges:
                graph.add_edge(source, target, edge_type=predicate, weight=weight)

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
            self.cursor.execute("SELECT COUNT(*) FROM nodes")
            stats['total_nodes'] = self.cursor.fetchone()[0]

            self.cursor.execute("SELECT node_type, COUNT(*) FROM nodes GROUP BY node_type")
            stats['nodes_by_type'] = dict(self.cursor.fetchall())

            # Edge counts
            self.cursor.execute("SELECT COUNT(*) FROM graph")
            stats['total_edges'] = self.cursor.fetchone()[0]

            self.cursor.execute("SELECT predicate, COUNT(*) FROM graph GROUP BY predicate")
            stats['edges_by_type'] = dict(self.cursor.fetchall())

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            raise

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
                query = """
                SELECT target, predicate, weight FROM graph
                WHERE source = %s AND predicate = %s
                UNION
                SELECT source, predicate, weight FROM graph
                WHERE target = %s AND predicate = %s
                LIMIT %s
                """
                self.cursor.execute(query, (node_id, edge_type, node_id, edge_type, max_neighbors))
            else:
                query = """
                SELECT target, predicate, weight FROM graph
                WHERE source = %s
                UNION
                SELECT source, predicate, weight FROM graph
                WHERE target = %s
                LIMIT %s
                """
                self.cursor.execute(query, (node_id, node_id, max_neighbors))

            return self.cursor.fetchall()

        except Exception as e:
            self.logger.error(f"Failed to query neighbors: {e}")
            raise
