"""Graph to Dictim converter - converts Graph instances to dictim string format"""

from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from collections import defaultdict
from src import Graph
import json


def _cluster_path(cluster_id: str, cluster_id_to_parent_id: Callable[[str], Optional[str]]) -> List[str]:
    """Build the full path from root to cluster."""
    def up(parts: List[str]) -> List[str]:
        if not parts:
            return parts
        
        parent_id = cluster_id_to_parent_id(parts[0])
        if parent_id is None or parent_id == parts[0]:  # prevent stack overflow
            return parts
        return up([parent_id] + parts)
    
    return up([cluster_id])


def _join_parts(parts: List[str]) -> str:
    """Join parts with dots to create qualified names."""
    return ".".join(str(p) for p in parts)


def _qualify_node(node_id: str, 
                  node_id_to_cluster_id: Callable[[str], Optional[str]], 
                  cluster_id_to_parent_id: Callable[[str], Optional[str]]) -> str:
    """Qualify a node to its place in the cluster hierarchy."""
    cluster_id = node_id_to_cluster_id(node_id)
    if cluster_id:
        path = _cluster_path(cluster_id, cluster_id_to_parent_id)
        return _join_parts(path + [node_id])
    return node_id


def _qualify_cluster(cluster_id: str, 
                     cluster_id_to_parent_id: Callable[[str], Optional[str]]) -> str:
    """Qualify a cluster to its place in the cluster hierarchy."""
    path = _cluster_path(cluster_id, cluster_id_to_parent_id)
    return _join_parts(path)


def _remove_nil(items: List[Any]) -> List[Any]:
    """Remove None values and empty collections."""
    return [item for item in items 
            if item is not None and not (isinstance(item, (list, dict)) and len(item) == 0)]


def _format_edge(src_key: str, dest_key: str, directed: bool, label: Optional[str], attrs: Dict,
                 context: Dict) -> List[Any]:
    """Format an edge for dictim output."""
    src_node = context.get('src_node', True)
    dest_node = context.get('dest_node', True)
    node_key_to_cluster = context.get('node_key_to_cluster', lambda x: None)
    cluster_to_parent = context.get('cluster_to_parent', lambda x: None)
    qualify = context.get('qualify', True)
    
    if qualify:
        qualified_src = (_qualify_node(src_key, node_key_to_cluster, cluster_to_parent) 
                        if src_node 
                        else _qualify_cluster(src_key, cluster_to_parent))
        
        qualified_dest = (_qualify_node(dest_key, node_key_to_cluster, cluster_to_parent) 
                         if dest_node 
                         else _qualify_cluster(dest_key, cluster_to_parent))
        
        parts = [qualified_src, "->" if directed else "--", qualified_dest, label, attrs if attrs else None]
    else:
        parts = [src_key, "->" if directed else "--", dest_key, label, attrs if attrs else None]
    
    return _remove_nil(parts)


def _format_node(key: str, attrs: Dict) -> List[Any]:
    """Format a node for dictim output."""
    label = attrs.get('label')
    node_attrs = {k: v for k, v in attrs.items() if k != 'label'}
    parts = [key, label, node_attrs if node_attrs else None]
    return _remove_nil(parts)


def _build_tree(edges: Set[Tuple[str, Optional[str]]]) -> Dict:
    """Build a tree structure from parent-child relationships."""
    def get_nodes_with_parent(parent):
        return [child for child, p in edges if p == parent]
    
    def build_subtree(parent):
        children = get_nodes_with_parent(parent)
        if children:
            return {child: build_subtree(child) for child in children}
        return None
    
    # Find root nodes (those with None parent or self-parent)
    roots = get_nodes_with_parent(None)
    if len(roots) == 1:
        return build_subtree(roots[0])
    elif len(roots) > 1:
        return {root: build_subtree(root) for root in roots}
    return {}


def _layout_nodes(node_to_key: Callable, cluster: Optional[str], 
                  cluster_to_nodes: Dict, node_to_attrs: Callable) -> List[List[Any]]:
    """Layout nodes for a given cluster."""
    if cluster not in cluster_to_nodes:
        return []
    
    return [_format_node(node_to_key(node), node_to_attrs(node)) 
            for node in cluster_to_nodes[cluster]]


def _layout_edges(edges: List, **kwargs) -> List[List[Any]]:
    """Layout all edges with proper qualification."""
    edge_to_src_key = kwargs.get('edge_to_src_key', lambda e: e.get('src', e[0] if isinstance(e, (list, tuple)) else None))
    edge_to_dest_key = kwargs.get('edge_to_dest_key', lambda e: e.get('dest', e[1] if isinstance(e, (list, tuple)) else None))
    edge_to_attrs = kwargs.get('edge_to_attrs', lambda e: {})
    node_to_key = kwargs.get('node_to_key', lambda x: x)
    is_node = kwargs.get('is_node', lambda x: True)
    node_key_to_cluster = kwargs.get('node_key_to_cluster', lambda x: None)
    cluster_to_parent = kwargs.get('cluster_to_parent', lambda x: None)
    qualify = kwargs.get('qualify', True)
    
    result = []
    for edge in edges:
        src = edge_to_src_key(edge)
        dest = edge_to_dest_key(edge)
        attrs = edge_to_attrs(edge) or {}
        directed = attrs.get('directed', True)
        label = attrs.get('label')
        
        # Remove special keys from attrs
        edge_attrs = {k: v for k, v in attrs.items() if k not in ['label', 'directed']}
        
        context = {
            'src_node': is_node(src),
            'dest_node': is_node(dest),
            'node_key_to_cluster': node_key_to_cluster,
            'cluster_to_parent': cluster_to_parent,
            'qualify': qualify
        }
        
        formatted = _format_edge(src, dest, directed, label, edge_attrs, context)
        result.append(formatted)
    
    return result


def _subgraphs(cluster: str, subtree: Optional[Dict], 
               cluster_to_attrs: Callable, cluster_to_nodes: Dict,
               node_to_key: Callable, node_to_attrs: Callable) -> List[Any]:
    """Generate subgraph/cluster layout."""
    attrs = cluster_to_attrs(cluster) if cluster_to_attrs else {}
    label = attrs.get('label')
    cluster_attrs = {k: v for k, v in attrs.items() if k != 'label'}
    
    result = []
    
    # Add cluster header
    cluster_parts = [cluster, label, cluster_attrs if cluster_attrs else None]
    result.extend(_remove_nil(cluster_parts))
    
    # Add nodes in this cluster
    result.extend(_layout_nodes(node_to_key, cluster, cluster_to_nodes, node_to_attrs))
    
    # Add subclusters
    if subtree:
        for subcluster, subsubtree in subtree.items():
            result.append(_subgraphs(subcluster, subsubtree, cluster_to_attrs, 
                                   cluster_to_nodes, node_to_key, node_to_attrs))
    
    return _remove_nil(result)


def _tree_edges(nodes: List[str], to_parent: Callable[[str], Optional[str]]) -> Set[Tuple[str, Optional[str]]]:
    """Generate tree edges from nodes and their parent function."""
    edges = set()
    processed = set()
    
    def add_path(node):
        if node in processed:
            return
        
        parent = to_parent(node)
        if parent != node:  # Avoid self-loops
            edges.add((node, parent))
            processed.add(node)
            if parent:
                add_path(parent)
        else:
            edges.add((node, None))
            processed.add(node)
    
    for node in nodes:
        add_path(node)
    
    return edges


def graph_to_dictim(graph, 
                   node_to_key: Callable = None,
                   node_to_attrs: Callable = None,
                   edge_to_src_key: Callable = None,
                   edge_to_dest_key: Callable = None,
                   edge_to_attrs: Callable = None,
                   node_to_cluster: Callable = None,
                   cluster_to_parent: Callable = None,
                   cluster_to_attrs: Callable = None,
                   qualify: bool = True,
                   directives: Optional[str] = None) -> str:
    """
    Convert a Graph instance to dictim string format.
    
    Args:
        graph: Graph instance
        node_to_key: Function to extract key from node (default: use node ID)
        node_to_attrs: Function to get node attributes (default: use graph.attrs)
        edge_to_src_key: Function to get source key from edge (default: edge['src'])
        edge_to_dest_key: Function to get dest key from edge (default: edge['dest'])
        edge_to_attrs: Function to get edge attributes (default: use graph.attrs)
        node_to_cluster: Function to assign nodes to clusters (default: no clusters)
        cluster_to_parent: Function to get cluster parent (default: no hierarchy)
        cluster_to_attrs: Function to get cluster attributes (default: empty)
        qualify: Whether to qualify node names with cluster paths (default: True)
        directives: Optional directives to prepend to output
    
    Returns:
        Dictim string representation of the graph
    """
    
    # Set up default functions
    nodes = graph.nodes()
    edges = graph.edges()
    
    if node_to_key is None:
        node_to_key = lambda node_id: node_id
    
    if node_to_attrs is None:
        node_to_attrs = lambda node_id: {}
    
    if edge_to_src_key is None:
        edge_to_src_key = lambda edge: edge.get('src')
    
    if edge_to_dest_key is None:
        edge_to_dest_key = lambda edge: edge.get('dest')
    
    if edge_to_attrs is None:
        edge_to_attrs = lambda edge: {}
    
    if node_to_cluster is None:
        node_to_cluster = lambda node_id: None
    
    if cluster_to_parent is None:
        cluster_to_parent = lambda cluster_id: None
    
    if cluster_to_attrs is None:
        cluster_to_attrs = lambda cluster_id: {}
    
    # Build cluster mappings
    cluster_to_nodes = defaultdict(list)
    for node_id in nodes:
        cluster = node_to_cluster(node_id)
        cluster_to_nodes[cluster].append(node_id)
    
    # Build cluster hierarchy
    cluster_tree = {}
    if any(cluster_to_nodes.keys()):
        clusters = [c for c in cluster_to_nodes.keys() if c is not None]
        if clusters:
            cluster_edges = _tree_edges(clusters, cluster_to_parent)
            cluster_tree = _build_tree(cluster_edges)
    
    # Helper functions
    node_keys = {node_to_key(node_id) for node_id in nodes}
    is_node = lambda key: key in node_keys
    
    def node_key_to_cluster(key):
        for node_id in nodes:
            if node_to_key(node_id) == key:
                return node_to_cluster(node_id)
        return None
    
    # Build the output
    result_parts = []
    
    # Add directives if provided
    if directives:
        if ":" in directives:
            key, value = directives.split(":", 1)
            directive_dict = {key.strip(): value.strip()}
            result_parts.append(directive_dict)
        else:
            result_parts.append(directives)
    
    # Layout clusters
    def add_cluster_tree(tree, level=0):
        if isinstance(tree, dict):
            for cluster, subtree in tree.items():
                cluster_layout = _subgraphs(cluster, subtree if isinstance(subtree, dict) else None,
                                          cluster_to_attrs, cluster_to_nodes,
                                          node_to_key, node_to_attrs)
                result_parts.extend(cluster_layout)
        
    if cluster_tree:
        add_cluster_tree(cluster_tree)
    
    # Layout nodes not in clusters
    unclustered_nodes = _layout_nodes(node_to_key, None, cluster_to_nodes, node_to_attrs)
    result_parts.extend(unclustered_nodes)
    
    # Layout all edges
    edge_layouts = _layout_edges(edges,
                                edge_to_src_key=edge_to_src_key,
                                edge_to_dest_key=edge_to_dest_key,
                                edge_to_attrs=edge_to_attrs,
                                node_to_key=node_to_key,
                                is_node=is_node,
                                node_key_to_cluster=node_key_to_cluster,
                                cluster_to_parent=cluster_to_parent,
                                qualify=qualify)
    result_parts.extend(edge_layouts)
    
    # Convert to string format
    def format_item(item):
        if isinstance(item, dict):
            return json.dumps(item)
        elif isinstance(item, list):
            return json.dumps(item)
        elif isinstance(item, str):
            return json.dumps([item])
        else:
            return json.dumps([str(item)])
    
    # Format all parts
    formatted_lines = []
    for part in result_parts:
        if isinstance(part, list) and any(isinstance(subpart, list) for subpart in part):
            # Nested structure (like clusters)
            for subpart in part:
                formatted_lines.append(format_item(subpart))
        else:
            formatted_lines.append(format_item(part))

    return json.dumps([json.loads(line) for line in formatted_lines], indent=2)


# VISUALIZATION
# Don't worry about this part too much - it's purpose is to provide a standard visualiation
# of the demo graph, but tweak away as required!

# String formatting utilities
def seq_to_str(s: List[Any]) -> str:
    """Convert a sequence to comma-separated string."""
    return ", ".join(to_str(item) for item in s)


# markdown line return
mdlr = "\n"


def md(s: str) -> str:
    """Wrap string in markdown format."""
    return f"|md {s} |"


def map_to_str(m: Dict[str, Any], tab: str = "") -> str:
    """Convert a map/dict to string representation with optional tabbing."""
    result_parts = []
    
    for k, v in m.items():
        key_name = k if isinstance(k, str) else str(k)
        
        if isinstance(v, (list, tuple, dict)):
            # Collection case
            result_parts.append(f"{mdlr}{tab}{key_name}:{mdlr}{to_str(v)}{tab}    ")
        else:
            # Simple value case
            result_parts.append(f"{key_name}: {to_str(v)}{mdlr}")
    
    return tab + "".join(result_parts)

def to_str(item: Any) -> str:
    """Convert any item to string representation."""
    if isinstance(item, dict):
        return map_to_str(item)
    elif isinstance(item, (list, tuple)):
        return seq_to_str(item)
    elif isinstance(item, (int, float)) and item < 0:
        return f"({abs(item)})"
    else:
        return str(item)

def get_nested(d: Dict, path: List[str], default=None):
    """Helper function to get nested dictionary value by path."""
    if not path:
        return None
    try:
        result = d
        for key in path:
            result = result[key]
        return result
    except (KeyError, TypeError):
        return default


def label_fn(attrs: Dict[str, Any], path: List[str]) -> str:
    """Generate label string from attributes."""
    name = attrs.get("name", "")
    cr = attrs.get("cr")
    
    result = f"{name}"

    if cr:
        result += f"{mdlr}{cr}"

    extra_data = get_nested(attrs, path)
        
    if extra_data:
        result += f"{mdlr}```\n{to_str(extra_data)}\n```"
    
    return result


def viz(g: Graph, *path: str) -> None:
    """
    Generate visualization dictim output and write to file.
    
    Args:
        g: Graph instance to visualize
        *path: Variable arguments forming the path for additional edge/node data
               to include in the visualization in a special ````code style```` block
               for instance, passing 'data' will add visualization of anything
               under the 'data' key of each node's attrs.
    """
    path_list = list(path) if path else None
    
    def node_to_attrs(node_id: str) -> Dict[str, Any]:
        """Convert node attributes to dictim attributes."""
        attrs = g.attrs(node_id) or {}
        
        # Standard node styling
        result = {
            "label": md(label_fn(attrs, path_list)),
            "shape": "rectangle",
            "style.border-radius": 1
        }
        
        # Add color if present - wrap hex colors in quotes
        if attrs.get("color"):
            color = attrs.get("color")
            if color.startswith("#"):
                result["style.fill"] = f"'{color}'"
            else:
                result["style.fill"] = color
                
        return result
    
    # Build the configuration for graph_to_dictim
    config = {
        "node_to_attrs": node_to_attrs,
        "directives": "direction: down",
        "qualify": True
    }
        
    # Generate dictim
    dictim_output = graph_to_dictim(g, **config)
    
    # Write to file instead of compiling to d2
    with open("hierarchy.json", "w") as f:
        f.write(dictim_output)
    
    print("Dictim output written to hierarchy.json")
