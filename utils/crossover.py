import random
from core.Genome.genome import Genome
from core.Node.node import NodeGene,ConnectionGene
def crossover(parent1: Genome, parent2: Genome) -> Genome:
    """ Crossover assuming parent1 is the fittest parent """
    offspring_connections = []
    offspring_nodes = set()
    all_nodes = {}  # Collect all nodes from both parents
    
    for node in parent1.nodes.values():
        all_nodes[node.id] = NodeGene(node.id, node.layer, node.activation, node.biais, node.aggregation)
        if node.layer in ("input", "output"):
            offspring_nodes.add(all_nodes[node.id]) # Ensure the input and output nodes are included
    for node in parent2.nodes.values():
        if node.id not in all_nodes:
            all_nodes[node.id] = NodeGene(node.id, node.layer, node.activation, node.biais, node.aggregation)      

    # Build maps of genes keyed by innovation number
    genes1 = {g.innov: g for g in parent1.connections}
    genes2 = {g.innov: g for g in parent2.connections}

    # Combine all innovation numbers
    all_innovs = set(genes1.keys()) | set(genes2.keys())

    for innov in sorted(all_innovs):
        gene1 = genes1.get(innov)
        gene2 = genes2.get(innov)
        
        if gene1 and gene2:  # Matching genes
            selected = random.choice([gene1, gene2])
            gene_copy = ConnectionGene(selected.in_node_id, selected.out_node_id, selected.weight, selected.innov, selected.enabled)

            if not gene1.enabled or not gene2.enabled:  # 75% chance of the offsprign gene being disabled
                if random.random() < 0.75:
                    gene_copy.enabled = False

        elif gene1 and not gene2:   # Disjoint gene (from the fittest parent)
            gene_copy = ConnectionGene(gene1.in_node_id, gene1.out_node_id, gene1.weight, gene1.innov, gene1.enabled)
        
        else:   # Not taking disjoint genes from less fit parent
            continue
        
        # get nodes
        in_node = all_nodes.get(gene_copy.in_node_id)
        out_node = all_nodes.get(gene_copy.out_node_id)
        
        if in_node and out_node:
            offspring_connections.append(gene_copy)
            offspring_nodes.add(in_node)
            offspring_nodes.add(out_node)
    
    offspring_nodes = list(offspring_nodes) # Remove the duplicates
    
    return Genome(offspring_nodes, offspring_connections)