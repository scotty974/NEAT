import numpy as np

# Fonctions d'agrégation pour NEAT
def sum_aggregation(inputs):
    """Somme de tous les inputs (par défaut)"""
    return sum(inputs) if inputs else 0.0

def product_aggregation(inputs):
    """Produit de tous les inputs"""
    if not inputs:
        return 0.0
    result = 1.0
    for x in inputs:
        result *= x
    return result

def max_aggregation(inputs):
    """Maximum des inputs"""
    return max(inputs) if inputs else 0.0

def min_aggregation(inputs):
    """Minimum des inputs"""
    return min(inputs) if inputs else 0.0

def mean_aggregation(inputs):
    """Moyenne des inputs"""
    return sum(inputs) / len(inputs) if inputs else 0.0

def median_aggregation(inputs):
    """Médiane des inputs"""
    if not inputs:
        return 0.0
    sorted_inputs = sorted(inputs)
    n = len(sorted_inputs)
    if n % 2 == 0:
        return (sorted_inputs[n//2 - 1] + sorted_inputs[n//2]) / 2
    return sorted_inputs[n//2]

def maxabs_aggregation(inputs):
    """Input avec la plus grande valeur absolue"""
    if not inputs:
        return 0.0
    return max(inputs, key=abs)

# Map des fonctions d'agrégation
aggregation_map = {
    "sum": sum_aggregation,
    "product": product_aggregation,
    "max": max_aggregation,
    "min": min_aggregation,
    "mean": mean_aggregation,
    "median": median_aggregation,
    "maxabs": maxabs_aggregation
}

# Fonction helper pour obtenir une agrégation
def get_aggregation(name):
    return aggregation_map.get(name, sum_aggregation)