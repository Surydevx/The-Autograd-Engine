import json
import os
from graphviz import Digraph

def render_graph(json_file="Model/telemetry.json", output_name="computational_graph"):
    """Reads telemetry JSON and renders a DAG using Graphviz."""
    
    if not os.path.exists(json_file):
        print(f"Error: Could not find {json_file}")
        return

    print(f"Loading graph data from {json_file}...")
    with open(json_file, 'r') as f:
        graph_data = json.load(f)

    nodes = graph_data.get('nodes', [])
    edges = graph_data.get('edges', [])
    
    print(f"Found {len(nodes)} nodes and {len(edges)} edges. Drawing...")
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

    for node in nodes:
        uid = node['id']
        op = node.get('label', '')
        data = node.get('data', 0.0)
        grad = node.get('grad', 0.0)

        if op:
            label = f"{{ {op} | data {data:.4f} | grad {grad:.4f} }}"
            color = "lightblue"
        else:
            label = f"{{ data {data:.4f} | grad {grad:.4f} }}"
            color = "white"

        dot.node(name=uid, label=label, shape='record', style='filled', fillcolor=color)

    for edge in edges:
        dot.edge(edge['from'], edge['to'])

    output_path = dot.render(output_name, view=True)
    print(f"Done! Graph saved and opened: {output_path}")

if __name__ == "__main__":
    render_graph("Model/telemetry.json")