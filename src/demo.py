"""
Demo file for tree walking examples using the simple-graph library.

Run sections by uncommenting the function calls at the bottom.
Each section builds on the previous ones, so run them in order.
"""

from src import Graph, graph_to_dictim, g1_prime, viz


def section_1():
    """Create and visualize the graph."""
    
    print("="*60)
    print("SECTION 1: Creating and visualizing g1_prime graph")
    print("="*60)

    # Create the graph from g1_prime data
    g = Graph(g1_prime, 'lei')
    
    print(f"Graph summary: {g.summary()}")
    print(f"Root nodes: {g.roots()}")
    print(f"Leaf nodes: {g.leaves()}")

    # Show some node details
    root_node = g.roots()[0]
    print(f"\nRoot node '{root_node}' details:")
    print(f"  Name: {g.attrs(root_node).get('name')}")
    print(f"  Level: {g.attrs(root_node).get('level')}")
    print(f"  Children: {g.children(root_node)}")

    # Visualize the original graph
    print(f"\nGenerating visualization...")
    viz(g)
    print("Check hierarchy.dict file to see the visualization data")
    
    return g

def section_2():
    """Add pink color to root node only."""
    
    print("\n" + "="*60)
    print("SECTION 2: Adding pink color to root node")
    print("="*60)

    def add_color_to_root():
        """Add pink color to the root node."""
        # Create fresh graph from g1_prime
        new_graph = Graph(g1_prime, 'lei')
        root_node = new_graph.roots()[0]
        root_attrs = new_graph.attrs(root_node).copy()
        root_attrs['color'] = 'pink'
        new_graph.set_node_attrs(root_node, root_attrs)
        return new_graph

    # Add color to root node
    g_with_root_color = add_color_to_root()
    root_node = g_with_root_color.roots()[0]
    print(f"Added pink color to root node: {g_with_root_color.attrs(root_node).get('name')}")

    # Check which nodes have color
    colored_nodes = [node for node in g_with_root_color.nodes() 
                     if g_with_root_color.attrs(node).get('color')]
    print(f"Nodes with color: {len(colored_nodes)} out of {g_with_root_color.node_count()}")

    # Visualize the graph with colored root
    print(f"\nGenerating visualization with colored root...")
    viz(g_with_root_color)
    print("Check hierarchy.dict file - only the root should be colored")
    
    return g_with_root_color

def section_3():
    """Prewalk the color down the tree."""
    
    print("\n" + "="*60)
    print("SECTION 3: Prewalking color inheritance down the tree")
    print("="*60)

    # Create fresh graph with pink root (no dependency on section 2)
    g_with_root_color = Graph(g1_prime, 'lei')
    root_node = g_with_root_color.roots()[0]
    root_attrs = g_with_root_color.attrs(root_node).copy()
    root_attrs['color'] = 'pink'
    g_with_root_color.set_node_attrs(root_node, root_attrs)
    
    print(f"Created graph with pink root: {g_with_root_color.attrs(root_node).get('name')}")

    def inherit_color_down(parent_attrs, current_attrs):
        """
        Prewalk modification function: inherit color from parent to child.
        
        Args:
            parent_attrs: Dictionary of parent node attributes
            current_attrs: Dictionary of current node attributes
        
        Returns:
            New attributes dictionary for current node
        """
        new_attrs = current_attrs.copy()
        
        # If parent has a color and current node doesn't, inherit it
        if parent_attrs.get('color') and not current_attrs.get('color'):
            new_attrs['color'] = parent_attrs['color']
            print(f"  Inheriting color '{parent_attrs['color']}' to {current_attrs.get('name', 'Unknown')}")
        
        return new_attrs

    # Prewalk to inherit color down the tree
    print("\nPrewalking color inheritance down the tree:")
    g_colored = g_with_root_color.prewalk_attrs(inherit_color_down)

    # Check results
    colored_nodes = [node for node in g_colored.nodes() 
                     if g_colored.attrs(node).get('color')]
    print(f"\nNodes with color: {len(colored_nodes)} out of {g_colored.node_count()}")

    # Show colored nodes by level
    for level in [0, 1, 2, 3]:
        level_nodes = [node for node in colored_nodes 
                       if g_colored.attrs(node).get('level') == level]
        if level_nodes:
            print(f"  Level {level}: {len(level_nodes)} nodes")

    # Visualize the fully colored tree
    print(f"\nGenerating fully colored tree visualization...")
    viz(g_colored)
    print("Check hierarchy.dict file - all nodes should now be colored pink")
    
    return g_colored

def section_4():
    """Conditional prewalking - don't overwrite existing colors."""
       
    print("\n" + "="*60)
    print("SECTION 4: Conditional color inheritance (preserve existing colors)")
    print("="*60)

    # Start fresh from g_with_root_color (red root only)
    # Add green color to specific node
    target_node = "LEI00012025z12026"
    g_mixed_colors = Graph(g1_prime, 'lei')
    
    # Add red to root
    root_node = g_mixed_colors.roots()[0]
    root_attrs = g_mixed_colors.attrs(root_node).copy()
    root_attrs['color'] = 'pink'
    g_mixed_colors.set_node_attrs(root_node, root_attrs)
    
    # Add green to target node
    target_attrs = g_mixed_colors.attrs(target_node).copy()
    target_attrs['color'] = 'green'
    g_mixed_colors.set_node_attrs(target_node, target_attrs)
    
    print(f"Set {g_mixed_colors.attrs(target_node).get('name')} to green color")
    print(f"Root {g_mixed_colors.attrs(root_node).get('name')} remains pink")

    def inherit_color_conditional(parent_attrs, current_attrs):
        """
        Conditional prewalk: only inherit color if current node has none.
        
        Args:
            parent_attrs: Dictionary of parent node attributes
            current_attrs: Dictionary of current node attributes
        
        Returns:
            New attributes dictionary for current node
        """
        new_attrs = current_attrs.copy()
        
        # Only inherit if parent has color AND current node has NO color
        if parent_attrs.get('color') and not current_attrs.get('color'):
            new_attrs['color'] = parent_attrs['color']
            print(f"  Inheriting {parent_attrs['color']} to {current_attrs.get('name', 'Unknown')}")
        elif current_attrs.get('color'):
            print(f"  Preserving {current_attrs['color']} on {current_attrs.get('name', 'Unknown')}")
        
        return new_attrs

    # Prewalk with conditional inheritance
    print("\nPrewalking with conditional color inheritance:")
    g_colored = g_mixed_colors.prewalk_attrs(inherit_color_conditional)

    # Check results by color
    pink_nodes = [node for node in g_colored.nodes() 
                 if g_colored.attrs(node).get('color') == 'pink']
    green_nodes = [node for node in g_colored.nodes() 
                   if g_colored.attrs(node).get('color') == 'green']
    
    print(f"\nFinal results:")
    print(f"  Pink nodes: {len(pink_nodes)}")
    print(f"  Green nodes: {len(green_nodes)}")
    print(f"  Total colored: {len(pink_nodes) + len(green_nodes)} out of {g_colored.node_count()}")

    # Visualize the mixed-color tree
    print(f"\nGenerating mixed-color tree visualization...")
    viz(g_colored)
    print("Check hierarchy.dict file - should show pink and green nodes!")
    
    return g_colored

def setup_section5_graph():
    g_with_data = Graph(g1_prime, 'lei')
    
    # Add nested data to root node
    root_node = g_with_data.roots()[0]
    root_attrs = g_with_data.attrs(root_node).copy()
    root_attrs['data'] = {
        'client_relationship': 'Markets',
        'sector': 'aerospace'
    }
    g_with_data.set_node_attrs(root_node, root_attrs)
    
    # Add nested data to MightyFunding
    mighty_funding_node = "LEI00012025z12026"  # MightyFunding's LEI
    mighty_attrs = g_with_data.attrs(mighty_funding_node).copy()
    mighty_attrs['data'] = {
        'client_relationship': 'commercial',
        'sector': 'investment'
    }
    g_with_data.set_node_attrs(mighty_funding_node, mighty_attrs)

    print(f"Added data to root node: {g_with_data.attrs(root_node).get('name')}")
    print(f"Root data: {root_attrs['data']}")
    print(f"Added data to: {g_with_data.attrs(mighty_funding_node).get('name')}")
    print(f"MightyFunding data: {mighty_attrs['data']}")

    return g_with_data


def section_5():
    """Add nested data to root node and MightyFunding, visualize with data path."""
        
    print("\n" + "="*60)
    print("SECTION 5: Adding nested data to root node and MightyFunding")
    print("="*60)

    g_with_data = setup_section5_graph()
    
    # Visualize with data path (will show the data in tooltips)
    print(f"\nGenerating visualization with data path...")
    viz(g_with_data, "data")
    print("Check hierarchy.dict file - data should now be visible in tooltips for both nodes!")
    
    return g_with_data

def section_6():
    """Conditionally prewalk sector data down the tree."""
    
    print("\n" + "="*60)
    print("SECTION 6: Conditionally prewalking sector data down the tree")
    print("="*60)

    g_with_data = setup_section5_graph()
    
    print(f"Initial setup complete - root has aerospace sector, MightyFunding has investment sector")

    def inherit_sector_conditional(parent_attrs, current_attrs):
        """
        Conditionally inherit sector from parent, but don't override if node has client_relationship.
        
        Args:
            parent_attrs: Dictionary of parent node attributes
            current_attrs: Dictionary of current node attributes
        
        Returns:
            New attributes dictionary for current node
        """
        new_attrs = current_attrs.copy()
        
        # Get parent and current data
        parent_data = parent_attrs.get('data', {})
        current_data = current_attrs.get('data', {})
        
        # If current node has client_relationship, don't override anything
        if current_data.get('client_relationship'):
            print(f"  Preserving data on {current_attrs.get('name', 'Unknown')} (has client_relationship)")
            return new_attrs
        
        # If parent has sector data and current node doesn't have any data, inherit sector
        if parent_data.get('sector') and not current_data:
            new_attrs['data'] = {'sector': parent_data['sector']}
            print(f"  Inheriting sector '{parent_data['sector']}' to {current_attrs.get('name', 'Unknown')}")
        
        return new_attrs

    # Prewalk with conditional sector inheritance
    print("\nPrewalking sector inheritance (conditional on client_relationship):")
    g_sector_walked = g_with_data.prewalk_attrs(inherit_sector_conditional)

    # Check results
    nodes_with_data = [node for node in g_sector_walked.nodes() 
                       if g_sector_walked.attrs(node).get('data')]
    
    print(f"\nNodes with data: {len(nodes_with_data)}")
    for node in nodes_with_data:
        attrs = g_sector_walked.attrs(node)
        data = attrs.get('data', {})
        print(f"  {attrs.get('name', 'Unknown')}: {data}")

    # Visualize the result
    print(f"\nGenerating visualization with inherited sector data...")
    viz(g_sector_walked, "data")
    print("Check hierarchy.dict file - aerospace sector should be inherited except where client_relationship exists!")
    
    return g_sector_walked

# hard code some revenues at the leaf nodes of our hierarchy
revenues = {
    'The Finance Nerds': 7,
    'Captal': 4,
    'Networth': 11,
    'Feminance': 14,
    'Prosperous Ledger': 1,
    'Calqulate': 12,
    'Finvac': 4,
    'Vacfin': 1,
    'FrostFinance': 8,
    'MoonFund': 20,
    'Investae': 5
}

def setup_graph_for_aggregation():
    print("\n" + "="*60)
    print("SECTION 7: Adding revenue data to nodes")
    print("="*60)

    # Start with the graph from section 5 setup    
    g_with_data = setup_section5_graph()
    
    print(f"Looking up revenue data for nodes...")
    
    # Add revenue data to nodes that have entries in the revenues dict
    nodes_with_revenue = 0
    for node_id in g_with_data.nodes():
        node_attrs = g_with_data.attrs(node_id)
        node_name = node_attrs.get('name')
        
        # Look up revenue for this node name
        if node_name in revenues:
            revenue_amount = revenues[node_name]
            
            # Get existing data or create new data dict
            current_data = node_attrs.get('data', {}).copy()
            current_data['revenue'] = revenue_amount
            
            # Update the node with new data
            new_attrs = node_attrs.copy()
            new_attrs['data'] = current_data
            g_with_data.set_node_attrs(node_id, new_attrs)
            
            print(f"  Added revenue ${revenue_amount}M to {node_name}")
            nodes_with_revenue += 1
    
    print(f"\nAdded revenue data to {nodes_with_revenue} nodes")

    # Show summary of nodes with data
    nodes_with_data = [node for node in g_with_data.nodes() 
                       if g_with_data.attrs(node).get('data')]
    
    print(f"\nNodes with data (showing all data):")
    for node in nodes_with_data:
        attrs = g_with_data.attrs(node)
        data = attrs.get('data', {})
        print(f"  {attrs.get('name', 'Unknown')}: {data}")    

    return g_with_data

def section_7():
    """Add revenue data to nodes by looking up in the revenues dictionary."""
    g_with_data = setup_graph_for_aggregation()
    
    # Visualize the graph with revenue data
    print(f"\nGenerating visualization with revenue data...")
    viz(g_with_data, "data")
    print("Check hierarchy.json file - nodes with revenue should show revenue amounts!")
    
    return g_with_data

def section_8():
    """Postwalk to aggregate revenue totals up the tree."""
    
    print("\n" + "="*60)
    print("SECTION 8: Postwalking revenue aggregation up the tree")
    print("="*60)

    # Start with the graph that has revenue data on leaf nodes
    g_with_revenue = setup_graph_for_aggregation()

    def aggregate_revenue_up(current_attrs, children_attrs_list):
        """Sum up revenue from children and add to current node."""
        new_attrs = current_attrs.copy()
        current_data = new_attrs.get('data', {}).copy()
        
        # Sum up revenue from all children
        children_total = sum(child_attrs.get('data', {}).get('revenue', 0) 
                           for child_attrs in children_attrs_list)
        
        # Add our own revenue (if any) to children's total
        own_revenue = current_data.get('revenue', 0)
        total_revenue = own_revenue + children_total
        
        # Set revenue if we have any
        if total_revenue > 0:
            current_data['revenue'] = total_revenue
            new_attrs['data'] = current_data
        
        return new_attrs

    # Postwalk to aggregate revenue up the tree
    print("Postwalking revenue aggregation...")
    g_aggregated = g_with_revenue.postwalk_attrs(aggregate_revenue_up)

    # Show the final root total
    root_node = g_aggregated.roots()[0]
    root_total = g_aggregated.attrs(root_node).get('data', {}).get('revenue', 0)
    print(f"Total company revenue: ${root_total}M")

    # Visualize the result
    viz(g_aggregated, "data")
    print("Check hierarchy.json file - all nodes should now show revenue!")
    
    return g_aggregated

# Color buckets for heatmap - mountain range style browns
# Heavily biased toward smaller values to show revenue origins
color_buckets = {
    0: "#f9f6f2",      # Lightest - valleys/low values (0-2%)
    2: "#f0e5da",      # Very light orange-brown (2-8%)
    8: "#e5ceb8",      # Light orange-brown (8-18%)
    18: "#d9b396",     # Medium light orange-brown (18-35%)
    35: "#cd9974",     # Medium orange-brown (35-55%)
    55: "#c18552",     # Medium dark orange-brown (55-75%)
    75: "#b67130"      # Darkest orange-brown - peaks/high values (75-100%)
}


def apply_heatmap_colors(graph, path):
    """
    Apply heatmap colors to tree nodes based on values at the given path.
    
    Args:
        graph: Graph instance
        path: List of keys to navigate to value (e.g., ['data', 'total_revenue'])
    
    Returns:
        New Graph with color attributes applied
    """
    
    def get_nested_value(attrs, path):
        """Extract value from nested path, return 0 if not found."""
        try:
            value = attrs
            for key in path:
                value = value[key]
            return value if isinstance(value, (int, float)) else 0
        except (KeyError, TypeError):
            return 0
    
    # Extract all values to find min/max
    values = []
    for node_id in graph.nodes():
        attrs = graph.attrs(node_id)
        value = get_nested_value(attrs, path)
        if value > 0:  # Only include nodes with actual values
            values.append(value)
    
    if not values:
        return graph  # No values to colorize
    
    min_val, max_val = min(values), max(values)
    value_range = max_val - min_val
    
    def get_color_for_value(value):
        """Map value to color bucket."""
        if value <= 0:
            return None  # No color for zero/negative values
        
        # Calculate percentage of range
        if value_range == 0:
            percentage = 50  # Middle color if all values are the same
        else:
            percentage = ((value - min_val) / value_range) * 100
        
        # Find appropriate color bucket
        for threshold in sorted(color_buckets.keys(), reverse=True):
            if percentage >= threshold:
                return color_buckets[threshold]
        return color_buckets[0]  # Default to lightest
    
    def add_heatmap_color(parent_attrs, current_attrs):
        """Prewalk function to add color based on value."""
        new_attrs = current_attrs.copy()
        value = get_nested_value(current_attrs, path)
        color = get_color_for_value(value)
        
        if color:
            new_attrs['color'] = color
        
        return new_attrs
    
    return graph.prewalk_attrs(add_heatmap_color)

def section_9():
    """Apply heatmap colors based on revenue values."""
    
    print("\n" + "="*60)
    print("SECTION 9: Applying heatmap colors based on revenue")
    print("="*60)

    # Start with aggregated revenue data
    g_aggregated = section_8()  # Get the tree with revenue on all nodes
    
    # Apply heatmap coloring based on revenue
    g_heatmap = apply_heatmap_colors(g_aggregated, ['data', 'revenue'])
    
    print("Applied revenue-based heatmap colors to the tree")
    
    # Visualize the heatmap
    viz(g_heatmap, "data")
    print("Check hierarchy.json file - all nodes should show brown heatmap colors!")
    
    return g_heatmap

def section_10():
    """Conditional aggregation with client relationship revenue offsets."""
    
    print("\n" + "="*60)
    print("SECTION 10: Conditional aggregation with client offsets")
    print("="*60)

    # Start with the graph that has revenue data on leaf nodes
    g_with_revenue = setup_graph_for_aggregation()

    def aggregate_with_client_offsets(current_attrs, children_attrs_list):
        """Sum revenue normally, but add revenue2_offset when current node has CR children."""
        new_attrs = current_attrs.copy()
        current_data = new_attrs.get('data', {}).copy()
        
        # Sum up revenue2 and revenue2_offset from all children
        # Check for both 'revenue2' (from aggregated nodes) and 'revenue' (from leaf nodes)
        children_revenue = sum(child_attrs.get('data', {}).get('revenue2', 0) or 
                              child_attrs.get('data', {}).get('revenue', 0)
                              for child_attrs in children_attrs_list)
        children_offset = sum(child_attrs.get('data', {}).get('revenue2_offset', 0) 
                             for child_attrs in children_attrs_list)
        
        # Only calculate revenue2 if we have children (i.e., we're above leaf nodes)
        if children_revenue > 0:
            # Get our own revenue if any (for nodes that have both children AND own revenue)
            own_revenue = current_data.get('revenue', 0)
            total_revenue = own_revenue + children_revenue
            current_data['revenue2'] = total_revenue
        
        # Check if any children have client_relationship - if so, create offset for THIS node
        cr_children_revenue = 0
        for child_attrs in children_attrs_list:
            child_data = child_attrs.get('data', {})
            if child_data.get('client_relationship'):
                # Get revenue from either revenue2 (aggregated) or revenue (leaf)
                cr_child_revenue = child_data.get('revenue2', 0) or child_data.get('revenue', 0)
                cr_children_revenue += cr_child_revenue
                print(f"  Found CR child: {child_attrs.get('name')} with ${cr_child_revenue}M")
        
        if cr_children_revenue > 0:
            current_data['revenue2_offset'] = -cr_children_revenue
            print(f"  Creating offset: {current_attrs.get('name')} gets revenue2_offset of ${-cr_children_revenue}M")
        
        # Add any existing revenue2_offset from children (propagate offsets up)
        if children_offset != 0:
            existing_offset = current_data.get('revenue2_offset', 0)
            current_data['revenue2_offset'] = existing_offset + children_offset
        
        # Update node data
        new_attrs['data'] = current_data
        return new_attrs

    # Postwalk with client relationship offsets
    print("Postwalking with client relationship offsets...")
    g_aggregated = g_with_revenue.postwalk_attrs(aggregate_with_client_offsets)

    # Show results
    root_node = g_aggregated.roots()[0]
    root_data = g_aggregated.attrs(root_node).get('data', {})
    root_revenue = root_data.get('revenue2', 0)
    root_offset = root_data.get('revenue2_offset', 0)
    net_revenue = root_revenue + root_offset
    
    print(f"\nFinal results:")
    print(f"  Total revenue2: ${root_revenue}M")
    print(f"  Client offsets: ${root_offset}M")
    print(f"  Net revenue: ${net_revenue}M")

    # Visualize the result
    viz(g_aggregated, "data")
    print("Check hierarchy.json file - should show revenue2 and revenue2_offset!")
    
    return g_aggregated

def section_11():
    """Winner takes all aggregation - CR nodes win the running total and reset to zero."""
    
    print("\n" + "="*60)
    print("SECTION 11: Client relationship winner takes all")
    print("="*60)

    # Start with the graph that has revenue data on leaf nodes
    g_with_revenue = setup_graph_for_aggregation()

    def aggregate_winner_takes_all(current_attrs, children_attrs_list):
        """Sum revenue, but CR nodes win the total and reset aggregation above them."""
        new_attrs = current_attrs.copy()
        current_data = new_attrs.get('data', {}).copy()
        
        # Sum up revenue3 from children (but not from CR children - they reset)
        children_revenue = 0
        for child_attrs in children_attrs_list:
            child_data = child_attrs.get('data', {})
            
            # If child is a CR node, it "wins" its revenue and we don't aggregate it up
            if child_data.get('client_relationship'):
                print(f"  CR node {child_attrs.get('name')} wins its revenue - not aggregating up")
                continue
            
            # Otherwise, sum up revenue3 (from aggregated nodes) or revenue (from leaf nodes)
            child_revenue = child_data.get('revenue3', 0) or child_data.get('revenue', 0)
            children_revenue += child_revenue
        
        # Only calculate revenue3 if we have children (i.e., we're above leaf/other nodes)
        if children_revenue > 0:
            # Get our own revenue if any (for nodes that have both children AND own revenue)
            own_revenue = current_data.get('revenue', 0)
            total_revenue = own_revenue + children_revenue
            current_data['revenue3'] = total_revenue
            print(f"  {current_attrs.get('name')} gets revenue3: ${total_revenue}M")
        
        # Update node data
        new_attrs['data'] = current_data
        return new_attrs

    # Postwalk with winner takes all logic
    print("Postwalking with winner takes all aggregation...")
    g_aggregated = g_with_revenue.postwalk_attrs(aggregate_winner_takes_all)

    # Show results
    root_node = g_aggregated.roots()[0]
    root_data = g_aggregated.attrs(root_node).get('data', {})
    root_revenue3 = root_data.get('revenue3', 0)
    
    print(f"\nFinal results:")
    print(f"  Root revenue3: ${root_revenue3}M (only non-CR revenue reaches the top)")

    # Show which nodes have revenue3
    nodes_with_revenue3 = []
    for node_id in g_aggregated.nodes():
        attrs = g_aggregated.attrs(node_id)
        data = attrs.get('data', {})
        if data.get('revenue3'):
            nodes_with_revenue3.append((attrs.get('name'), data.get('revenue3')))
    
    print(f"\nNodes with revenue3:")
    for name, revenue3 in nodes_with_revenue3:
        print(f"  {name}: ${revenue3}M")

    # Visualize the result
    viz(g_aggregated, "data")
    print("Check hierarchy.json file - should show revenue3 where appropriate!")
    
    return g_aggregated


# ==============================================================================
# Run sections by uncommenting the calls below
# ==============================================================================

if __name__ == "__main__":
    # Uncomment the sections you want to run:
    # section_1()
    # section_2()
    # section_3()
    # section_4()
    
    # section_5()
    # section_6()

    # section_7()
    # section_8()
    # section_9()
    # section_10()
    section_11()

