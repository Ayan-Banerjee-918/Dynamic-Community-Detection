import random
import networkx as nx
import pandas as pd
from bokeh.plotting import figure, curdoc
from bokeh.models.sources import ColumnDataSource
from bokeh.palettes import Category10

# Function to initialize and update communities
def fluidc_dynamic_weighted_visualization(doc, csv_path, k, max_iter=100, seed=0):
    random.seed(seed)

    def initialize_communities(G, k):
        return {node: i % k for i, node in enumerate(G.nodes)}

    def initialize_densities(communities, G):
        return {i: 1 / sum(G[node][neighbor].get("weight", 1) for neighbor in G.neighbors(node)) for node in G.nodes for i in set(communities.values())}

    def update_densities(communities, G):
        return {i: 1 / max(1, sum(G[node][neighbor].get("weight", 1) for neighbor in G.neighbors(node))) for node in G.nodes for i in set(communities.values())}

    def calculate_modularity(G, communities):
        modularity = 0
        m = G.size(weight='weight')

        for i in set(communities.values()):
            community_nodes = [node for node in communities if communities[node] == i]
            subgraph = G.subgraph(community_nodes)
            L_in = subgraph.size(weight='weight')
            k_in = sum(G.degree(weight='weight')[node] for node in community_nodes)
            modularity += (L_in / m) - (k_in / (2 * m)) ** 2

        return modularity

    best_partition = None
    best_num_non_empty_communities = 0
    best_modularity = float('-inf')

    df = pd.read_csv(csv_path)

    # Create a graph with all nodes and edges
    G = nx.Graph()
    G.add_edges_from(df[['Source', 'Target']].values)

    # Create a Bokeh plot
    plot = figure(title="Fluid Communities Visualization", x_range=(-1.5, 1.5), y_range=(-1.5, 1.5), tools="")
    source = ColumnDataSource(data={"x": [], "y": [], "color": []})
    communities_renderer = plot.circle(x="x", y="y", size=10, color="color", alpha=0.6, source=source)

    # Function to update the plot
    def update():
        nonlocal best_partition, best_num_non_empty_communities, best_modularity

        communities = initialize_communities(G, k)
        densities = initialize_densities(communities, G)

        iter_count = 0
        converged = False

        while not converged and iter_count < max_iter:
            iter_count += 1
            nodes = list(G.nodes)
            random.shuffle(nodes)
            changes = 0

            for v in nodes:
                sums = {
                    i: sum(
                        densities[communities[n]] * G[v][n].get("weight", 1) for n in G.neighbors(v) if communities[n] == i
                    )
                    for i in set(communities.values())
                }
                c = max(sums, key=sums.get)

                if communities[v] != c:
                    changes += 1
                    densities[c] = 1 / sum(G[v][n].get("weight", 1) for n in G.neighbors(v))
                    densities[communities[v]] = update_densities(communities, G)[communities[v]]
                    communities[v] = c

            if changes == 0:
                converged = True

        non_empty_communities = {i: {node for node in communities if communities[node] == i} for i in set(communities.values())}
        num_non_empty_communities = sum(1 for community in non_empty_communities.values() if len(community) > 0)
        modularity = calculate_modularity(G, communities)

        if num_non_empty_communities == k and modularity > best_modularity:
            best_partition = non_empty_communities
            best_num_non_empty_communities = num_non_empty_communities
            best_modularity = modularity

        if num_non_empty_communities > best_num_non_empty_communities and modularity > best_modularity:
            best_partition = non_empty_communities
            best_num_non_empty_communities = num_non_empty_communities
            best_modularity = modularity

        # Smooth transition between node positions
        pos = nx.spring_layout(G, seed=42)  # Seed for consistent layout
        target_pos = {node: [pos[node][0] + random.uniform(-0.05, 0.05), pos[node][1] + random.uniform(-0.05, 0.05)] for node in G.nodes}

        # Update the Bokeh plot data and title
        source.data = {"x": [pos[node][0] for node in G.nodes],
                       "y": [pos[node][1] for node in G.nodes],
                       "color": [Category10[10][color % 10] for color in communities.values()]}
        plot.title.text = f"Fluid Communities Visualization - Timestep {update.i}"

        # Update node positions for the next timestep
        nx.set_node_attributes(G, target_pos, 'pos')

        # Increment the timestep index
        update.i += 1

    update.i = 0  # Initialize the timestep index

    # Periodically update the plot
    doc.add_periodic_callback(update, 500)  # Adjust the callback interval as needed

    # Display the plot
    doc.add_root(plot)

# Usage example:

fluidc_dynamic_weighted_visualization(curdoc(), 'hcw_edges.csv', 2)