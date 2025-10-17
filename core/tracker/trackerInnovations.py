class InovationsTracker:
    def __init__(self):
        self.current_inovation = 0
        self.connection_innovations = {}
        self.node_innovations = {}
        self.node_id_counter = 0

    #  on recupere l'innovation d'une connection
    def get_connection_innovation(self, in_node_id, out_node_id):
        # on cree la key
        key = (in_node_id, out_node_id)
        #  on check si la clés existe déjà ou pas dans le dictionnaire
        if key not in self.connection_innovations:
            #  si elle n'existe pas on l'ajoute
            self.connection_innovations[key] = self.current_inovation
            self.current_inovation += 1
        # on retourne l'innovation
        return self.connection_innovations[key]
    
    # on recupere l'innovation d'un noeud
    def get_node_innovation(self, connection_innovation):
        # on check si le noeud existe deja
        if connection_innovation not in self.node_innovations:
            # si il n'existe pas on l'ajoute
            node_id = self.node_id_counter
            self.node_id_counter += 1
            # on cree l'innovation
            conn1_innov = self.current_inovation
            self.current_inovation += 1
            # on cree l'innovation
            conn2_innov = self.current_inovation
            self.current_inovation += 1
            # on ajoute l'innovation
            self.node_innovations[connection_innovation] = (node_id, conn1_innov, conn2_innov)
        return self.node_innovations[connection_innovation]