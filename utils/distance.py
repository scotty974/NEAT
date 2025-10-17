from core.Genome.genome import Genome

def distance(genome1: Genome, genome2: Genome, c1=1.0, c2=1.0, c3=0.4):
    # Création de dictionnaires de gènes par innovation
    genes1 = {g.innov: g for g in genome1.connections}
    genes2 = {g.innov: g for g in genome2.connections}
    
    innovations1 = set(genes1.keys())
    innovations2 = set(genes2.keys())
    
    # Matching, disjoint, excess
    matching = innovations1 & innovations2
    disjoint = (innovations1 ^ innovations2)
    
    max_innov1 = max(innovations1) if innovations1 else 0
    max_innov2 = max(innovations2) if innovations2 else 0
    
    excess = set()
    # Définir correctement les gènes excess par rapport aux maxima individuels
    for innov in disjoint.copy():
        if innov > max(max_innov1, max_innov2):
            excess.add(innov)
            disjoint.remove(innov)
    
    # Différence moyenne de poids pour les gènes matching
    if matching:
        weight_diff = sum(abs(genes1[i].weight - genes2[i].weight) for i in matching)
        avg_weight_diff = weight_diff / len(matching)
    else:
        avg_weight_diff = 0.0

    # Normalisation par le nombre de gènes
    N = max(len(genes1), len(genes2))
    if N < 20:
        N = 1

    delta = c1 * len(excess) / N + c2 * avg_weight_diff + c3 * len(disjoint) / N
    return delta
