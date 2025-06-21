"""graph data structures"""

import uuid
from typing import Dict, List, Any, Optional

# READING SECTION
# This is how the user will specify a graph. With a simple dict of nodes and edges.
d1 = {"nodes": [{"lei": "1234z",
                 "name": "Guru Funds"},
                {"lei": "1235z",
                 "name": "Binance"},
                {"lei": "1236z",
                 "name": "Steadfast money"}],
      "edges": [{"src": "1234z", "dest": "1235z"},
                {"src": "1235z", "dest": "1236z", "color": "red"}]}

# note:
#   `nodes` are specified as a list of dicts. The keys in each dict are arbitrary, but
#   one key must be the same and guaranteed to be unique. 'lei' in our example.
#   `attrs` are also specified as a list of dicts. Each edge must be 'src' & 'dest'
#   keys. Beyond that they can have whatever the user wants to specify.


# The internal representation of the graph is like this. Much more efficient
# for bidirectional edge traversal operations needed to support graph algorithms.
# at the cost of some data repetition
# Also note that each edge is given a unique id. This is because we want to be able
# to support more than one edge between the same two nodes.
# the `node_key` for nodes is `lei` in our example
# the `edge_key` for edges is always the `id`
g1 = {"nodemap": {'1234z':
                  {'out_edges':
                   {'1235z': [{'src': '1234z',
                               'dest': '1235z',
                               'id': '54f8128b-3ca6-409c-82d4-48c04f509a73'}]},
                   'in_edges': {}},
                  '1235z':
                  {'out_edges':
                   {'1236z': [{'src': '1235z',
                               'dest': '1236z',
                               'color': 'red',
                               'id': '2545a6f7-6052-45a4-92aa-9a82b3673705'}]},
                   'in_edges':
                   {'1234z': [{'src': '1234z',
                               'dest': '1235z',
                               'id': '54f8128b-3ca6-409c-82d4-48c04f509a73'}]}},
                  '1236z':
                  {'out_edges': {},
                   'in_edges':
                   {'1235z': [{'src': '1235z',
                               'dest': '1236z',
                               'id': '2545a6f7-6052-45a4-92aa-9a82b3673705'}]}}},
      "attrs": {"1234z":
                {"lei": "1234z",
                 "name": "Guru Funds"},
                "1235z":
                {"lei": "1235z",
                 "name": "Binance"},
                "1236z":
                {"lei": "1236z",
                 "name": "Steadfast money"},
                '54f8128b-3ca6-409c-82d4-48c04f509a73':
                {'src': '1234z',
                 'dest': '1235z',
                 'id': '54f8128b-3ca6-409c-82d4-48c04f509a73'},
                '2545a6f7-6052-45a4-92aa-9a82b3673705':
                {'src': '1235z',
                 'dest': '1236z',
                 'color': 'red',
                 'id': '2545a6f7-6052-45a4-92aa-9a82b3673705'}}}
# END SECTION


# converts the first data structure to the second
def _init_graph(graph_data, node_key):
    """
    Convert a graph from simple format to efficient search format.
    
    Args:
        graph_data: Dict with 'nodes' and 'edges' lists
        node_key: String key used for node uniqueness (e.g., 'lei')
    
    Returns:
        Dict with 'nodemap' and 'attrs' for efficient graph operations
    """
    # Initialize result structure
    result = {
        "nodemap": {},
        "attrs": {}
    }
    
    # Process nodes - create nodemap entries and store attributes
    for node in graph_data.get('nodes', []):
        node_id = node[node_key]
        result["nodemap"][node_id] = {
            "out_edges": {},
            "in_edges": {}
        }
        result["attrs"][node_id] = node
    
    # Process edges - populate in_edges and out_edges
    for edge in graph_data.get('edges', []):
        # Add unique ID to edge
        edge_id = str(uuid.uuid4())
        edge_with_id = {**edge, "id": edge_id}
        
        # Store full edge in attrs map
        result["attrs"][edge_id] = edge_with_id
        
        # Create minimal edge reference for nodemap
        edge_ref = {"src": edge['src'], "dest": edge['dest'], "id": edge_id}
        
        src = edge['src']
        dest = edge['dest']
        
        # Add to out_edges of source node
        if dest not in result["nodemap"][src]["out_edges"]:
            result["nodemap"][src]["out_edges"][dest] = []
        result["nodemap"][src]["out_edges"][dest].append(edge_ref)
        
        # Add to in_edges of destination node
        if src not in result["nodemap"][dest]["in_edges"]:
            result["nodemap"][dest]["in_edges"][src] = []
        result["nodemap"][dest]["in_edges"][src].append(edge_ref)
    
    return result


class Graph:
    """
    A graph data structure for learning and experimentation.
    
    Supports directed graphs with arbitrary attributes on nodes and edges.
    Optimized for fast traversal operations.
    """
    
    def __init__(self, graph_data=None, node_key="id"):
        """
        Initialize a Graph with efficient internal representation.
        
        Args:
            graph_data: Dict with 'nodes' and 'edges' lists, or None for empty graph
            node_key: String key used for node uniqueness (e.g., 'lei', 'id')
        """
        self._node_key = node_key
        
        if graph_data is None:
            # Create empty graph
            self._nodemap = {}
            self._attrs = {}
        else:
            result = _init_graph(graph_data, node_key)
            self._nodemap = result["nodemap"]
            self._attrs = result["attrs"]
    
    # === BASIC ACCESS ===
    
    def nodes(self) -> List[str]:
        """Get all node primary keys."""
        return list(self._nodemap.keys())
    
    def edges(self) -> List[Dict]:
        """Get all edges with their full attribute maps."""
        edges = []
        for edge_id, edge_attrs in self._attrs.items():
            # Check if this is an edge (has 'src' and 'dest' keys)
            if 'src' in edge_attrs and 'dest' in edge_attrs:
                edges.append(edge_attrs)
        return edges
    
    def attrs(self, key: str) -> Optional[Dict]:
        """Get the attribute map for a node or edge."""
        return self._attrs.get(key)
    
    def node_count(self) -> int:
        """Get the number of nodes in the graph."""
        return len(self._nodemap)
    
    def edge_count(self) -> int:
        """Get the number of edges in the graph."""
        return len([e for e in self._attrs.values() if 'src' in e and 'dest' in e])
    
    # === MODIFICATION OPERATIONS ===
    
    def add_node(self, node_id: str, attrs: Dict = None) -> None:
        """
        Add a node to the graph.
        
        Args:
            node_id: Unique identifier for the node
            attrs: Optional dictionary of node attributes
        """
        if node_id in self._nodemap:
            raise ValueError(f"Node {node_id} already exists")
        
        self._nodemap[node_id] = {
            "out_edges": {},
            "in_edges": {}
        }
        
        # Create attrs dict with the node_key
        node_attrs = attrs.copy() if attrs else {}
        node_attrs[self._node_key] = node_id
        self._attrs[node_id] = node_attrs
    
    def remove_node(self, node_id: str) -> None:
        """
        Remove a node and all connected edges from the graph.
        
        Args:
            node_id: Node identifier to remove
        """
        if node_id not in self._nodemap:
            raise ValueError(f"Node {node_id} does not exist")
        
        # Remove all edges connected to this node
        edges_to_remove = []
        for edge in self.edges():
            if edge['src'] == node_id or edge['dest'] == node_id:
                edges_to_remove.append(edge['id'])
        
        for edge_id in edges_to_remove:
            self.remove_edge_by_id(edge_id)
        
        # Remove the node itself
        del self._nodemap[node_id]
        del self._attrs[node_id]
    
    def add_edge(self, src: str, dest: str, attrs: Dict = None) -> str:
        """
        Add an edge to the graph.
        
        Args:
            src: Source node identifier
            dest: Destination node identifier
            attrs: Optional dictionary of edge attributes
            
        Returns:
            The generated edge ID
        """
        if src not in self._nodemap:
            raise ValueError(f"Source node {src} does not exist")
        if dest not in self._nodemap:
            raise ValueError(f"Destination node {dest} does not exist")
        
        # Generate edge ID and create edge
        edge_id = str(uuid.uuid4())
        edge_attrs = attrs.copy() if attrs else {}
        edge_attrs.update({"src": src, "dest": dest, "id": edge_id})
        
        # Store full edge in attrs
        self._attrs[edge_id] = edge_attrs
        
        # Create minimal edge reference for nodemap
        edge_ref = {"src": src, "dest": dest, "id": edge_id}
        
        # Add to out_edges of source node
        if dest not in self._nodemap[src]["out_edges"]:
            self._nodemap[src]["out_edges"][dest] = []
        self._nodemap[src]["out_edges"][dest].append(edge_ref)
        
        # Add to in_edges of destination node
        if src not in self._nodemap[dest]["in_edges"]:
            self._nodemap[dest]["in_edges"][src] = []
        self._nodemap[dest]["in_edges"][src].append(edge_ref)
        
        return edge_id
    
    def remove_edge(self, src: str, dest: str, edge_id: str = None) -> None:
        """
        Remove an edge from the graph.
        
        Args:
            src: Source node identifier
            dest: Destination node identifier
            edge_id: Optional specific edge ID (if multiple edges between same nodes)
        """
        if src not in self._nodemap or dest not in self._nodemap:
            raise ValueError("Source or destination node does not exist")
        
        # Find and remove from out_edges
        if dest in self._nodemap[src]["out_edges"]:
            edges = self._nodemap[src]["out_edges"][dest]
            if edge_id:
                edges[:] = [e for e in edges if e["id"] != edge_id]
                removed_id = edge_id
            else:
                if edges:
                    removed_id = edges[0]["id"]
                    edges.pop(0)
                else:
                    raise ValueError(f"No edge from {src} to {dest}")
            
            # Clean up empty lists
            if not edges:
                del self._nodemap[src]["out_edges"][dest]
        else:
            raise ValueError(f"No edge from {src} to {dest}")
        
        # Find and remove from in_edges
        if src in self._nodemap[dest]["in_edges"]:
            edges = self._nodemap[dest]["in_edges"][src]
            edges[:] = [e for e in edges if e["id"] != removed_id]
            
            # Clean up empty lists
            if not edges:
                del self._nodemap[dest]["in_edges"][src]
        
        # Remove from attrs
        if removed_id in self._attrs:
            del self._attrs[removed_id]
    
    def remove_edge_by_id(self, edge_id: str) -> None:
        """
        Remove an edge by its ID.
        
        Args:
            edge_id: Edge identifier to remove
        """
        if edge_id not in self._attrs:
            raise ValueError(f"Edge {edge_id} does not exist")
        
        edge = self._attrs[edge_id]
        if 'src' not in edge or 'dest' not in edge:
            raise ValueError(f"Invalid edge {edge_id}")
        
        self.remove_edge(edge['src'], edge['dest'], edge_id)
    
    def set_node_attrs(self, node_id: str, attrs: Dict) -> None:
        """
        Replace the entire attribute map for a node.
        
        Args:
            node_id: Node identifier
            attrs: New attribute dictionary
        """
        if node_id not in self._nodemap:
            raise ValueError(f"Node {node_id} does not exist")
        
        # Ensure the node_key is preserved
        new_attrs = attrs.copy()
        new_attrs[self._node_key] = node_id
        self._attrs[node_id] = new_attrs
    
    def set_edge_attrs(self, edge_id: str, attrs: Dict) -> None:
        """
        Replace the entire attribute map for an edge.
        
        Args:
            edge_id: Edge identifier
            attrs: New attribute dictionary
        """
        if edge_id not in self._attrs:
            raise ValueError(f"Edge {edge_id} does not exist")
        
        old_edge = self._attrs[edge_id]
        if 'src' not in old_edge or 'dest' not in old_edge:
            raise ValueError(f"Invalid edge {edge_id}")
        
        # Ensure required keys are preserved
        new_attrs = attrs.copy()
        new_attrs.update({
            "src": old_edge["src"],
            "dest": old_edge["dest"], 
            "id": edge_id
        })
        self._attrs[edge_id] = new_attrs
    
    # === TRAVERSAL ===
    
    def children(self, node: str) -> List[str]:
        """Get child nodes of a given node."""
        if node not in self._nodemap:
            return []
        return list(self._nodemap[node]["out_edges"].keys())
    
    def parents(self, node: str) -> List[str]:
        """Get parent nodes of a given node."""
        if node not in self._nodemap:
            return []
        return list(self._nodemap[node]["in_edges"].keys())
    
    def roots(self) -> List[str]:
        """Get all root node keys (nodes with no parents)."""
        return [node for node in self._nodemap 
                if len(self._nodemap[node]["in_edges"]) == 0]
    
    def leaves(self) -> List[str]:
        """Get all leaf nodes (nodes with no children)."""
        return [node for node in self._nodemap 
                if len(self._nodemap[node]["out_edges"]) == 0]
    
    def descendants(self, node: str, max_depth: Optional[int] = None) -> List[str]:
        """Get all descendants of a node using BFS."""
        if node not in self._nodemap:
            return []
        
        visited = set()
        queue = [(node, 0)]
        result = []
        
        while queue:
            current, depth = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            if current != node:  # Don't include the starting node
                result.append(current)
            
            if max_depth is None or depth < max_depth:
                for child in self.children(current):
                    if child not in visited:
                        queue.append((child, depth + 1))
        
        return result
    
    def ancestors(self, node: str, max_depth: Optional[int] = None) -> List[str]:
        """Get all ancestors of a node using BFS."""
        if node not in self._nodemap:
            return []
        
        visited = set()
        queue = [(node, 0)]
        result = []
        
        while queue:
            current, depth = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            if current != node:  # Don't include the starting node
                result.append(current)
            
            if max_depth is None or depth < max_depth:
                for parent in self.parents(current):
                    if parent not in visited:
                        queue.append((parent, depth + 1))
        
        return result
    
    # === ANALYSIS ===
    
    def tree(self) -> bool:
        """Check if the graph is a strict hierarchy/tree."""
        node_list = self.nodes()
        
        # Empty graph or single node is a tree
        if len(node_list) <= 1:
            return True
        
        # Count nodes with exactly zero parents (should be exactly 1 root)
        # and nodes with exactly one parent (should be all others)
        roots = 0
        for node in node_list:
            parent_count = len(self._nodemap[node]["in_edges"])
            if parent_count == 0:
                roots += 1
            elif parent_count != 1:
                return False  # Node has multiple parents
        
        # Must have exactly one root
        return roots == 1
    
    def is_dag(self) -> bool:
        """Check if graph is a Directed Acyclic Graph using Kahn's algorithm."""
        in_degree = {node: len(self._nodemap[node]["in_edges"]) 
                    for node in self._nodemap}
        queue = [node for node, degree in in_degree.items() if degree == 0]
        processed = 0
        
        while queue:
            node = queue.pop(0)
            processed += 1
            
            for child in self.children(node):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        
        return processed == len(self._nodemap)
    
    def connected_components(self) -> List[List[str]]:
        """Find all connected components (treating graph as undirected)."""
        visited = set()
        components = []
        
        def dfs(node: str, component: List[str]):
            if node in visited:
                return
            visited.add(node)
            component.append(node)
            
            # Visit both children and parents (undirected)
            for neighbor in self.children(node) + self.parents(node):
                dfs(neighbor, component)
        
        for node in self._nodemap:
            if node not in visited:
                component = []
                dfs(node, component)
                components.append(component)
        
        return components
    
    # === ALGORITHMS ===
    
    def shortest_path(self, start: str, end: str) -> Optional[List[str]]:
        """Find shortest path between two nodes using BFS."""
        if start not in self._nodemap or end not in self._nodemap:
            return None
        
        if start == end:
            return [start]
        
        queue = [(start, [start])]
        visited = set()
        
        while queue:
            node, path = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            
            for child in self.children(node):
                new_path = path + [child]
                if child == end:
                    return new_path
                if child not in visited:
                    queue.append((child, new_path))
        
        return None
    
    def topological_sort(self) -> Optional[List[str]]:
        """Return topologically sorted nodes (if DAG)."""
        if not self.is_dag():
            return None
        
        in_degree = {node: len(self._nodemap[node]["in_edges"]) 
                    for node in self._nodemap}
        queue = [node for node, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            for child in self.children(node):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        
        return result
    
    def has_path(self, start: str, end: str) -> bool:
        """Check if there's a path from start to end node."""
        return self.shortest_path(start, end) is not None
    
    # === UTILITY ===
    
    def summary(self) -> str:
        """Get a summary string of the graph."""
        return (f"Graph: {self.node_count()} nodes, {self.edge_count()} edges, "
                f"Tree: {self.tree()}, DAG: {self.is_dag()}, "
                f"Components: {len(self.connected_components())}")
    
    def __str__(self) -> str:
        """String representation of the graph."""
        return self.summary()
    
    def __repr__(self) -> str:
        """Detailed representation of the graph."""
        return f"Graph(nodes={self.node_count()}, edges={self.edge_count()})"

    # === TREE WALKING ===
    
    def prewalk_tree(self, start_node: str, f) -> 'Graph':
        """
        Prewalk the tree starting at node, applying function f to update node attributes.
        
        Args:
            start_node: Node to start the walk from
            f: Function that takes (graph, parent_node, current_node) and returns new attrs
               for the current node
        
        Returns:
            New Graph with updated attributes
        
        Raises:
            ValueError: If graph is not a tree
        """
        if not self.tree():
            raise ValueError("Graph must be a tree for prewalk_tree")
        
        if start_node not in self._nodemap:
            raise ValueError(f"Start node {start_node} does not exist")
        
        # Create a copy of the current graph
        new_graph = Graph()
        new_graph._node_key = self._node_key
        new_graph._nodemap = {}
        new_graph._attrs = {}
        
        # Copy all nodes and edges to new graph
        for node_id in self.nodes():
            new_graph._nodemap[node_id] = {
                "out_edges": {},
                "in_edges": {}
            }
            new_graph._attrs[node_id] = self._attrs[node_id].copy()
        
        for edge in self.edges():
            edge_id = edge['id']
            new_graph._attrs[edge_id] = edge.copy()
            
            src, dest = edge['src'], edge['dest']
            edge_ref = {"src": src, "dest": dest, "id": edge_id}
            
            if dest not in new_graph._nodemap[src]["out_edges"]:
                new_graph._nodemap[src]["out_edges"][dest] = []
            new_graph._nodemap[src]["out_edges"][dest].append(edge_ref)
            
            if src not in new_graph._nodemap[dest]["in_edges"]:
                new_graph._nodemap[dest]["in_edges"][src] = []
            new_graph._nodemap[dest]["in_edges"][src].append(edge_ref)
        
        def down(graph, parent_node, current_node):
            # Apply function to get new attributes
            print(current_node)
            new_attrs = f(graph, parent_node, current_node)
            
            # Update the node's attributes using the existing method
            if new_attrs is not None:
                graph.set_node_attrs(current_node, new_attrs)
            
            # Recursively process children
            children = graph.children(current_node)
            for child in children:
                down(graph, current_node, child)
            
            return graph
        
        return down(new_graph, start_node, start_node)
    
    def prewalk_attrs(self, step_fn) -> 'Graph':
        """
        Prewalk the tree starting at the root, applying step_fn to node attributes.
        
        Args:
            step_fn: Function that takes (parent_attrs, current_attrs) and returns new_attrs
        
        Returns:
            New Graph with updated attributes
        """
        roots = self.roots()
        if not roots:
            raise ValueError("Graph has no roots")
        if len(roots) > 1:
            raise ValueError("Graph has multiple roots - use prewalk_tree for specific start node")
        
        start_node = roots[0]
        
        def wrapper_func(graph, parent_node, current_node):
            parent_attrs = graph.attrs(parent_node)
            current_attrs = graph.attrs(current_node)
            return step_fn(parent_attrs, current_attrs)
        
        return self.prewalk_tree(start_node, wrapper_func)
    
    def postwalk_tree(self, start_node: str, f) -> 'Graph':
        """
        Postwalk the tree starting at node, applying function f to update node attributes.
        
        Args:
            start_node: Node to start the walk from
            f: Function that takes (graph, current_node, child_nodes) and returns new attrs
               for the current node
        
        Returns:
            New Graph with updated attributes
        
        Raises:
            ValueError: If graph is not a tree
        """
        if not self.tree():
            raise ValueError("Graph must be a tree for postwalk_tree")
        
        if start_node not in self._nodemap:
            raise ValueError(f"Start node {start_node} does not exist")
        
        # Create a copy of the current graph
        new_graph = Graph()
        new_graph._node_key = self._node_key
        new_graph._nodemap = {}
        new_graph._attrs = {}
        
        # Copy all nodes and edges to new graph
        for node_id in self.nodes():
            new_graph._nodemap[node_id] = {
                "out_edges": {},
                "in_edges": {}
            }
            new_graph._attrs[node_id] = self._attrs[node_id].copy()
        
        for edge in self.edges():
            edge_id = edge['id']
            new_graph._attrs[edge_id] = edge.copy()
            
            src, dest = edge['src'], edge['dest']
            edge_ref = {"src": src, "dest": dest, "id": edge_id}
            
            if dest not in new_graph._nodemap[src]["out_edges"]:
                new_graph._nodemap[src]["out_edges"][dest] = []
            new_graph._nodemap[src]["out_edges"][dest].append(edge_ref)
            
            if src not in new_graph._nodemap[dest]["in_edges"]:
                new_graph._nodemap[dest]["in_edges"][src] = []
            new_graph._nodemap[dest]["in_edges"][src].append(edge_ref)
        
        def up(graph, current_node):
            # First, recursively process all children (post-order)
            children = graph.children(current_node)
            for child in children:
                up(graph, child)
            
            # Then apply function to current node
            new_attrs = f(graph, current_node, children)
            
            # Update the node's attributes using the existing method
            if new_attrs is not None:
                graph.set_node_attrs(current_node, new_attrs)
            
            return graph
        
        return up(new_graph, start_node)
    
    def postwalk_attrs(self, step_fn) -> 'Graph':
        """
        Postwalk the tree starting at the root, applying step_fn to node attributes.
        
        Args:
            step_fn: Function that takes (current_attrs, children_attrs_list) and returns new_attrs
        
        Returns:
            New Graph with updated attributes
        """
        roots = self.roots()
        if not roots:
            raise ValueError("Graph has no roots")
        if len(roots) > 1:
            raise ValueError("Graph has multiple roots - use postwalk_tree for specific start node")
        
        start_node = roots[0]
        
        def wrapper_func(graph, current_node, child_nodes):
            current_attrs = graph.attrs(current_node)
            children_attrs = [graph.attrs(child) for child in child_nodes]
            return step_fn(current_attrs, children_attrs)
        
        return self.postwalk_tree(start_node, wrapper_func)
