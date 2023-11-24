from bokeh.plotting import figure, show
from bokeh.models import HoverTool, ColumnDataSource, Slider, Button
from bokeh.models.widgets import Div
from bokeh.io import curdoc
from bokeh.layouts import column, row, Spacer
from bokeh.palettes import Category20_20
import networkx as nx
import pandas as pd
from networkx.algorithms import community
from cdlib.algorithms import leiden
from bokeh.events import DocumentReady

# Read CSV file
csv_file = 'hcw_edges.csv'
df = pd.read_csv(csv_file)

# Create a directed graph to represent temporal edges
G = nx.DiGraph()
G.add_edges_from(df[['Source', 'Target']].values)

pos = nx.kamada_kawai_layout(G)

# Create Bokeh plot
plot = figure(title="Community Structure", tools="pan,wheel_zoom,save,reset, tap", active_scroll='wheel_zoom')

# Initialize empty data sources for edges and nodes
edge_source = ColumnDataSource(data={'x0': [], 'y0': [], 'x1': [], 'y1': []})
node_source = ColumnDataSource(data={'x': [], 'y': [], 'index': [], 'community': [], 'fill_color': [], 'size': [], 'centrality': [],'in_degree': [],'out_degree': []})

# Add edges to the plot
edge_renderer = plot.segment(x0="x0", y0="y0", x1="x1", y1="y1", source=edge_source,
                            line_width=1, line_color="gray", line_alpha=0.6)

# Add nodes to the plot with community-specific colors, influence-based size, and hover
node_renderer = plot.circle(x='x', y='y', size='size', source=node_source, line_color='black',
                            fill_color='fill_color', legend_field='community')

# Add hover for nodes
hover_nodes = HoverTool(renderers=[node_renderer],
                        tooltips=[("Node", "@index"), ("Community", "@community"), ("Degree", "@size"), ("Centrality", "@centrality"), ("In-Degree", "@in_degree"), ("Out-Degree", "@out_degree")])

plot.add_tools(hover_nodes)

# Create a Div widget for displaying modularity
modularity_text = Div(text="<p style='font-size:110%'>Modularity: N/A</p>")

# Create a Div widget for displaying modularity
density_text = Div(text="<p style='font-size:110%'>Density: N/A</p>")

# Create a Div widget for displaying modularity
community_count_text = Div(text="<p style='font-size:110%'>Community Count: N/A</p>")

# Create a Div widget for displaying modularity
node_count_text = Div(text="<p style='font-size:110%'>Number of Nodes: N/A</p>")

edge_count_text = Div(text="<p style='font-size:110%'>Number of Edges: N/A</p>")

# Create a spacer for better layout control
spacer = Spacer(width=10, height=10)
# Create a Div widget for the title "Analytics"
analytics_title = Div(text="<h3>Analytics</h3>", styles={'margin-bottom': '5px'})

# Combine the title, modularity display, and border in a column layout
analytics_box = column(
    modularity_text,
    density_text,
    community_count_text,
    node_count_text,
    edge_count_text,
    width=200,  # Adjust the width according to your preference
    styles={'border': '2px solid #428bca', 'border-radius': '5px', 'padding': '10px', 'background-color': '#f2f2f2'}
)

communities_dict = {}
G_dict = {}

# Add edges and nodes for each timestep
def update_plot(_attr, _old, _new):
    global pos, communities_dict, G_dict
    timestep = slider.value
    
    if timestep not in G_dict:
        # Clear the graph for the new timestep
        G = nx.DiGraph()

        # Add time-stamped edges to the graph for the current timestep
        for _,row in df.iterrows():
            if row['time_start'] <= timestep:
                if G.has_edge(row['Source'],row['Target']):
                    G.edges[row['Source'],row['Target']]['weight'] += 1
                else:
                    G.add_edge(row['Source'], row['Target'],weight=1)

        G_dict[timestep]=G
    else:
        G=G_dict[timestep]
            
    # Check if communities for the current timestep are already computed
    if timestep not in communities_dict:
        # Perform community detection (Louvain method)
        #communities = leiden(G.to_undirected(),weights='weight').communities
        communities=nx.community.louvain_communities(G,weight='weight',seed=42)
        # Store communities in the dictionary
        communities_dict[timestep] = communities
    else:
        # Use stored communities for the current timestep
        communities = communities_dict[timestep]

    modularity = community.modularity(G, communities)
    density = nx.density(G)
    degree_centrality = nx.degree_centrality(G)
    in_degree_centrality = nx.in_degree_centrality(G)
    out_degree_centrality= nx.out_degree_centrality(G)
    
    # Update the modularity_text widget
    modularity_text.text = f"<p style='font-size:110%'>Modularity: {modularity:.4f}</p>"
    density_text.text = f"<p style='font-size:110%'>Density: {density:.4f}</p>"
    community_count_text.text = f"<p style='font-size:110%'>Community Count: {len(communities)}</p>"
    node_count_text.text = f"<p style='font-size:110%'>Number of Nodes: {len(G.nodes())}</p>"
    edge_count_text.text = f"<p style='font-size:110%'>Number of Edges: {len(G.edges())}</p>"

    # Extract community information for each node
    node_community_dict = {node: i for i, comm in enumerate(communities) for node in comm}

    # Get node degrees (influence) and normalize for node size
    node_degrees = dict(G.degree())
    max_degree = max(node_degrees.values())
    node_sizes = {node: 10 + 15 * (node_degrees[node] / max_degree) for node in G.nodes()}

    # Update the pos dictionary to center the graph in the plot
    center_x = sum([coord[0] for coord in pos.values()]) / len(pos)
    center_y = sum([coord[1] for coord in pos.values()]) / len(pos)

    # Offset the positions to center the graph
    pos = {node: (x - center_x, y - center_y) for node, (x, y) in pos.items()}

    # Update edge data source
    edge_source.data = {'x0': [], 'y0': [], 'x1': [], 'y1': []}
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_source.data['x0'].append(x0)
        edge_source.data['y0'].append(y0)
        edge_source.data['x1'].append(x1)
        edge_source.data['y1'].append(y1)

    # Update node data source
    node_source.data = {'x': [], 'y': [], 'index': [], 'community': [], 'fill_color': [], 'size': [], 'centrality':[], 'in_degree':[], 'out_degree':[]}

    for node in G.nodes():
        x, y = pos[node]
        community_color = Category20_20[node_community_dict[node] % len(Category20_20)]
        node_source.data['x'].append(x)
        node_source.data['y'].append(y)
        node_source.data['index'].append(node)
        node_source.data['community'].append(node_community_dict[node])
        node_source.data['fill_color'].append(community_color)
        node_source.data['size'].append(node_sizes[node])
        node_source.data['centrality'].append(degree_centrality[node])
        node_source.data['in_degree'].append(in_degree_centrality[node])
        node_source.data['out_degree'].append(out_degree_centrality[node])
    

    # Update renderers
    edge_renderer.data_source.data = dict(edge_source.data)
    node_renderer.data_source.data = dict(node_source.data)

def __init__(event):
    update_plot(None,None,None)

# Create a slider
slider = Slider(start=df['time_start'].min(), end=df['time_end'].max(),
                value=df['time_start'].min(), step=1, title="Timestep")
slider.on_change('value_throttled', update_plot)


animation_step_slider = Slider(start=1, end=100, value=1, step=1, title="Animation Step")

def animate_update():
    timestep=slider.value+animation_step_slider.value
    if(timestep>=df['time_end'].max()):
        timestep=1
    slider.value=timestep
    __init__(None)


callback_id = None

def animate():
    global callback_id
    if button.label == '► Play':
        button.label = '❚❚ Pause'
        button.button_type = 'danger'
        callback_id = curdoc().add_periodic_callback(animate_update, 1000)
    else:
        button.label = '► Play'
        button.button_type = 'success'
        curdoc().remove_periodic_callback(callback_id)

button = Button(label='► Play', width=60, button_type = 'success', align='center')
button.on_click(animate)    

# Create a slider for the step input
step_slider = Slider(start=1, end=100, value=1, step=1, title="Step Size")

# Function to update the main slider step based on the step input slider
def update_step(attr, old, new):
    global slider
    slider.step = step_slider.value
    

# Attach the update_step function to the step_slider's on_change event
step_slider.on_change('value', update_step)

# Create a layout with both sliders
slider_layout = row(slider, step_slider)

animation_layout = row(animation_step_slider, button)

# Create layout
layout = column(slider_layout, animation_layout, row(plot, column(row(spacer,analytics_title),row(spacer,analytics_box))))

# Set up the document
curdoc().add_root(layout)
curdoc().title = "Temporal Community Structure"
curdoc().on_event(DocumentReady, __init__)
