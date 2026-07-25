import json
import os
import argparse
from graphviz import Digraph
from collections import defaultdict, deque

def trim_graph_bfs(nodes, edges, max_depth=3):
    """Trims the graph backwards from the final loss node using BFS."""
    print(f"Trimming graph to a maximum depth of {max_depth}...")
    
    reverse_adj = defaultdict(list)
    out_degree = defaultdict(int)
    
    node_map = {n['id']: n for n in nodes}
    
    for edge in edges:
        u, v = edge['from'], edge['to']
        reverse_adj[v].append(u)
        out_degree[u] += 1
        if v not in out_degree:
            out_degree[v] = 0  
            
    roots = [nid for nid, deg in out_degree.items() if deg == 0]
    if not roots:
        roots = [nodes[-1]['id']] if nodes else []

    visited = set(roots)
    queue = deque([(root, 0) for root in roots])
    
    trimmed_node_ids = set(roots)
    trimmed_edges = []

    while queue:
        current, depth = queue.popleft()
        
        if depth < max_depth:
            for parent in reverse_adj[current]:
                trimmed_edges.append({'from': parent, 'to': current})
                trimmed_node_ids.add(parent)
                
                if parent not in visited:
                    visited.add(parent)
                    queue.append((parent, depth + 1))

    trimmed_nodes = [node_map[nid] for nid in trimmed_node_ids if nid in node_map]
    
    return trimmed_nodes, trimmed_edges


def render_graph(json_file, output_name, max_depth):
    """Reads telemetry JSON, trims it, and renders a DAG using Graphviz."""
    
    if not os.path.exists(json_file):
        print(f"[!] Error: Could not find {json_file}")
        return

    print(f"Loading graph data from {json_file}...")
    with open(json_file, 'r') as f:
        graph_data = json.load(f)

    raw_nodes = graph_data.get('nodes', [])
    raw_edges = graph_data.get('edges', [])
    print(f"Loaded raw graph: {len(raw_nodes)} nodes and {len(raw_edges)} edges.")
    
    nodes, edges = trim_graph_bfs(raw_nodes, raw_edges, max_depth=max_depth)
    
    print(f"Trimmed graph down to: {len(nodes)} nodes and {len(edges)} edges. Drawing...")
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

    for node in nodes:
        uid = str(node['id'])
        op = node.get('label', '')
        
        try:
            data = float(node.get('data') or 0.0)
            grad = float(node.get('grad') or 0.0)
        except (ValueError, TypeError):
            data, grad = 0.0, 0.0

        if op:
            label = f"{{ {op} | data {data:.4f} | grad {grad:.4f} }}"
            color = "lightblue"
        else:
            label = f"{{ data {data:.4f} | grad {grad:.4f} }}"
            color = "white"

        dot.node(name=uid, label=label, shape='record', style='filled', fillcolor=color)

    for edge in edges:
        dot.edge(str(edge['from']), str(edge['to']))
    
    try:
        output_path = dot.render(output_name, view=True)
        print(f"Done! Graph saved and opened: {output_path}")
    except Exception as e:
        print(f"[!] Graphviz rendering failed. Is the Graphviz system binary installed? Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autograd Telemetry Visualizer")
    parser.add_argument("--file", type=str, default="Model/telemetry.json", help="Path to telemetry.json")
    parser.add_argument("--out", type=str, default="Model/computational_graph", help="Output filename (without extension)")
    parser.add_argument("--depth", type=int, default=4, help="Max backwards depth to traverse from the Loss node")
    
    args = parser.parse_args()

    output_dir = os.path.dirname(args.out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    render_graph(args.file, args.out, args.depth)