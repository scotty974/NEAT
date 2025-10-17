import networkx as nx
import matplotlib.pyplot as plt

def plot_genome(genome):
    G = nx.DiGraph()
    
    # Ajout des noeuds
    for node in genome.nodes.values():
        if node.layer == "input":
            G.add_node(node.id, label=f"I{node.id}", layer=0, color='lightblue')
        elif node.layer == "output":
            G.add_node(node.id, label=f"O{node.id}", layer=2, color='lightgreen')
        else:
            G.add_node(node.id, label=f"H{node.id}", layer=1, color='orange')
    
    # Ajout des connexions
    for conn in genome.connections:
        if conn.enabled:
            G.add_edge(conn.in_node_id, conn.out_node_id, weight=conn.weight)

    pos = {}
    # Placement par couche
    layers = {0: [], 1: [], 2: []}
    for n, d in G.nodes(data=True):
        layers[d['layer']].append(n)
    
    for layer, nodes in layers.items():
        for i, n in enumerate(nodes):
            pos[n] = (layer, i)
    
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    
    node_colors = [d['color'] for _, d in G.nodes(data=True)]
    nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'), node_color=node_colors, node_size=800)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()
