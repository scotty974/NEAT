# fonction pour trier le reseau de node
def topological_sort(edges):
    visited = set()
    order = []
    
    def visit(n):
        if n in visited:
            return
        visited.add(n)
        for m in edges[n]:
            visit(m)
        order.append(n)
    
    for node in edges:
        visit(node)
    
    return order[::-1]