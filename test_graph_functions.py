"""
Test file for new graph traversal functions.

Tests df_traverse_attrs_from_start, df_traverse_attrs_from_starts, and df_trace_from_start
with a simple test graph.
"""

from src import Graph


def create_test_graph():
    """Create a simple test graph for demonstrating the new functions."""
    test_data = {
        "nodes": [
            {"id": "A", "name": "Node A", "value": 10, "level": 0},
            {"id": "B", "name": "Node B", "value": 20, "level": 1},
            {"id": "C", "name": "Node C", "value": 30, "level": 1},
            {"id": "D", "name": "Node D", "value": 40, "level": 2},
            {"id": "E", "name": "Node E", "value": 50, "level": 2},
            {"id": "F", "name": "Node F", "value": 60, "level": 2}
        ],
        "edges": [
            {"src": "A", "dest": "B", "weight": 5, "type": "normal"},
            {"src": "A", "dest": "C", "weight": 3, "type": "normal"},
            {"src": "B", "dest": "D", "weight": 2, "type": "heavy"},
            {"src": "B", "dest": "E", "weight": 4, "type": "normal"},
            {"src": "C", "dest": "F", "weight": 1, "type": "light"},
            {"src": "C", "dest": "E", "weight": 6, "type": "heavy"}
        ]
    }
    return Graph(test_data, "id")


def test_df_traverse_attrs_from_start():
    """Test the df_traverse_attrs_from_start function."""
    print("=" * 60)
    print("TESTING: df_traverse_attrs_from_start")
    print("=" * 60)
    
    g = create_test_graph()
    
    print("Original graph structure:")
    print(f"Nodes: {g.nodes()}")
    print(f"Root: {g.roots()}")
    print(f"Leaves: {g.leaves()}")
    
    # Test function: Add a "visited" flag to nodes as we traverse down
    def mark_as_visited(first_attrs, second_attrs):
        """Mark nodes as visited during traversal."""
        new_attrs = second_attrs.copy()
        new_attrs['visited'] = True
        new_attrs['visited_from'] = first_attrs.get('name', 'unknown')
        print(f"  Marking {second_attrs.get('name')} as visited from {first_attrs.get('name')}")
        return new_attrs
    
    print(f"\nTraversing DOWN from root 'A':")
    result_graph = g.df_traverse_attrs_from_start("down", "A", mark_as_visited)
    
    print(f"\nResults after traversal:")
    for node_id in result_graph.nodes():
        attrs = result_graph.attrs(node_id)
        visited = attrs.get('visited', False)
        visited_from = attrs.get('visited_from', 'N/A')
        print(f"  {attrs.get('name')}: visited={visited}, from={visited_from}")
    
    return result_graph


def test_df_traverse_attrs_from_starts():
    """Test the df_traverse_attrs_from_starts function."""
    print("\n" + "=" * 60)
    print("TESTING: df_traverse_attrs_from_starts")
    print("=" * 60)
    
    g = create_test_graph()
    
    # Test function: Add depth information as we traverse up from leaves
    def add_depth_info(first_attrs, second_attrs):
        """Add depth information when traversing up."""
        new_attrs = second_attrs.copy()
        first_depth = first_attrs.get('depth_from_leaf', 0)
        new_attrs['depth_from_leaf'] = first_depth + 1
        print(f"  Setting depth {first_depth + 1} for {second_attrs.get('name')} (from {first_attrs.get('name')})")
        return new_attrs
    
    leaves = g.leaves()
    print(f"Traversing UP from leaves: {leaves}")
    
    # Initialize depth for leaves
    g_with_leaf_depth = g
    for leaf in leaves:
        leaf_attrs = g_with_leaf_depth.attrs(leaf).copy()
        leaf_attrs['depth_from_leaf'] = 0
        g_with_leaf_depth.set_node_attrs(leaf, leaf_attrs)
        print(f"  Initialized {leaf_attrs.get('name')} with depth 0")
    
    result_graph = g_with_leaf_depth.df_traverse_attrs_from_starts("up", leaves, add_depth_info)
    
    print(f"\nResults after multi-start traversal:")
    for node_id in result_graph.nodes():
        attrs = result_graph.attrs(node_id)
        depth = attrs.get('depth_from_leaf', 'N/A')
        print(f"  {attrs.get('name')}: depth_from_leaf={depth}")
    
    return result_graph


def test_df_trace_from_start():
    """Test the df_trace_from_start function."""
    print("\n" + "=" * 60)
    print("TESTING: df_trace_from_start")
    print("=" * 60)
    
    g = create_test_graph()
    
    print("Original graph edges with weights:")
    for edge in g.edges():
        print(f"  {edge['src']} -> {edge['dest']}: weight={edge.get('weight')}, type={edge.get('type')}")
    
    # Edge filter: Only follow 'heavy' edges
    def follow_heavy_edges(accumulated_data, edges):
        """Only follow edges marked as 'heavy'."""
        heavy_edges = [e for e in edges if e.get('type') == 'heavy']
        edge_desc = [f"{e['src']}->{e['dest']}({e.get('type')})" for e in heavy_edges]
        print(f"  Filtering edges. Available: {len(edges)}, Heavy: {len(heavy_edges)} {edge_desc}")
        return heavy_edges
    
    # Collection function: Build path with total weight
    def collect_path_and_weight(accumulated_data, node_attrs, edge_attrs=None):
        """Collect path information and cumulative weight."""
        if accumulated_data is None:
            # Starting node
            path_data = {
                'path': [node_attrs.get('name')],
                'total_weight': 0,
                'total_value': node_attrs.get('value', 0)
            }
        else:
            # Add current node to path
            edge_weight = edge_attrs.get('weight', 0) if edge_attrs else 0
            path_data = {
                'path': accumulated_data['path'] + [node_attrs.get('name')],
                'total_weight': accumulated_data['total_weight'] + edge_weight,
                'total_value': accumulated_data['total_value'] + node_attrs.get('value', 0)
            }
        
        print(f"    Visiting {node_attrs.get('name')}: path={path_data['path']}, weight={path_data['total_weight']}")
        return path_data
    
    print(f"\nTracing from 'A' following only heavy edges:")
    final_data = g.df_trace_from_start("down", "A", follow_heavy_edges, collect_path_and_weight, 
                                      consumes_edge_attrs=True, initial_data=None)
    
    print(f"\nFinal trace results:")
    if final_data:
        print(f"  Final path: {' -> '.join(final_data['path'])}")
        print(f"  Total weight: {final_data['total_weight']}")
        print(f"  Total value: {final_data['total_value']}")
    else:
        print("  No valid path found")
    
    # Test 2: Follow normal edges
    print(f"\nTracing from 'A' following only normal edges:")
    
    def follow_normal_edges(accumulated_data, edges):
        """Only follow edges marked as 'normal'."""
        normal_edges = [e for e in edges if e.get('type') == 'normal']
        edge_desc = [f"{e['src']}->{e['dest']}({e.get('type')})" for e in normal_edges]
        print(f"  Filtering edges. Available: {len(edges)}, Normal: {len(normal_edges)} {edge_desc}")
        return normal_edges
    
    final_data_normal = g.df_trace_from_start("down", "A", follow_normal_edges, collect_path_and_weight,
                                             consumes_edge_attrs=True, initial_data=None)
    
    print(f"\nFinal trace results (normal edges):")
    if final_data_normal:
        print(f"  Final path: {' -> '.join(final_data_normal['path'])}")
        print(f"  Total weight: {final_data_normal['total_weight']}")
        print(f"  Total value: {final_data_normal['total_value']}")
    else:
        print("  No valid path found")
    
    return final_data


def main():
    """Run all tests."""
    print("Testing new graph traversal functions")
    print("Graph structure: A -> B,C -> D,E,F (with various edge types and weights)")
    
    test_df_traverse_attrs_from_start()
    test_df_traverse_attrs_from_starts() 
    test_df_trace_from_start()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()