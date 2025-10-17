import networkx as nx
import matplotlib.pyplot as plt

def plot_genome(genome, config=None):
    """
    Visualise un genome NEAT.
    
    Args:
        genome: Le genome NEAT à visualiser
        config: La configuration NEAT (optionnel, permet d'identifier les couches)
    """
    G = nx.DiGraph()
    
    # Déterminer les IDs des nœuds d'entrée et de sortie
    if config:
        input_ids = list(range(-config.genome_config.num_inputs, 0))
        output_ids = list(range(config.genome_config.num_outputs))
    else:
        # Heuristique : les IDs négatifs sont des entrées, les petits positifs sont des sorties
        all_node_ids = list(genome.nodes.keys())
        input_ids = [nid for nid in all_node_ids if nid < 0]
        # Supposer que les sorties sont les plus petits IDs positifs
        positive_ids = sorted([nid for nid in all_node_ids if nid >= 0])
        output_ids = positive_ids[:4] if len(positive_ids) >= 4 else positive_ids
    
    # Ajouter TOUS les nœuds d'entrée (même s'ils ne sont pas dans genome.nodes)
    for node_id in input_ids:
        G.add_node(node_id, label=f"I{node_id}", layer=0, color='lightblue')
    
    # Ajouter TOUS les nœuds de sortie (même s'ils ne sont pas dans genome.nodes)
    for node_id in output_ids:
        G.add_node(node_id, label=f"O{node_id}", layer=2, color='lightgreen')
    
    # Ajouter les nœuds cachés (qui sont dans genome.nodes)
    # Palette de couleurs pour les nœuds cachés
    hidden_colors = ['orange', 'yellow', 'coral', 'gold', 'peachpuff', 
                     'lightsalmon', 'sandybrown', 'moccasin', 'wheat', 'khaki']
    hidden_nodes = [nid for nid in genome.nodes.keys() 
                    if nid not in input_ids and nid not in output_ids]
    
    for i, node_id in enumerate(hidden_nodes):
        color = hidden_colors[i % len(hidden_colors)]
        G.add_node(node_id, label=f"H{node_id}", layer=1, color=color)
    
    # Ajout des connexions
    for conn_key, conn in genome.connections.items():
        if conn.enabled:
            in_node, out_node = conn_key
            G.add_edge(in_node, out_node, weight=conn.weight)
    
    # Placement par couche
    pos = {}
    layers = {0: [], 1: [], 2: []}
    
    # Collecter les nœuds par couche
    for n in G.nodes():
        node_data = G.nodes[n]
        layers[node_data['layer']].append(n)
    
    # Espacer verticalement les nœuds dans chaque couche
    for layer, nodes in layers.items():
        nodes_sorted = sorted(nodes)
        for i, n in enumerate(nodes_sorted):
            y_pos = i - len(nodes_sorted) / 2
            pos[n] = (layer * 3, y_pos)  # Multiplier par 3 pour plus d'espace horizontal
    
    # Préparation de l'affichage
    plt.figure(figsize=(12, 8))
    
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    node_colors = [G.nodes[n]['color'] for n in G.nodes()]
    node_labels = {n: G.nodes[n]['label'] for n in G.nodes()}
    
    # Dessiner le graphe
    nx.draw(G, pos, 
            with_labels=True, 
            labels=node_labels,
            node_color=node_colors, 
            node_size=1000,
            font_size=10,
            font_weight='bold',
            arrows=True,
            arrowsize=20,
            edge_color='gray',
            width=2)
    
    nx.draw_networkx_edge_labels(G, pos, 
                                  edge_labels=edge_labels,
                                  font_size=8)
    
    plt.title(f"Genome Visualization - Fitness: {genome.fitness:.2f}" if hasattr(genome, 'fitness') else "Genome Visualization",
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()