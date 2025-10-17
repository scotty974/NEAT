class NodeGene():
    def __init__(self, id, layer, activation, biais, aggregation):
        self.id = id
        self.layer = layer # type od node
        self.activation = activation #activate function
        self.biais = biais
        self.aggregation =  aggregation
        
    

    def __str__(self):
        return f"NodeGene(id={self.id}, layer={self.layer}, activation={self.activation}, biais={self.biais}, aggregation={self.aggregation})"
    def __repr__(self):
        return self.__str__()
class ConnectionGene():
    def __init__(self, in_node_id, out_node_id, weight, innov, enabled=True):
        self.in_node_id = in_node_id
        self.out_node_id = out_node_id
        self.weight = weight
        self.innov = innov
        self.enabled = enabled
        
    
    def __str__(self):
        return f"ConnectionGene(in_node_id={self.in_node_id}, out_node_id={self.out_node_id}, weight={self.weight}, innov={self.innov}, enabled={self.enabled})"
    
    def __repr__(self):
        return self.__str__()
    
    