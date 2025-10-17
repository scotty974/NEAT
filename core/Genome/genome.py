from utils.topologies import topological_sort
from core.tracker.trackerInnovations import InovationsTracker
from core.Node.node import NodeGene, ConnectionGene
import random
from utils.activations import activation_map
from utils.math import aggregation_map
import warnings


class Genome:
    def __init__(self, nodes, connections):
        self.nodes = {nodes.id: nodes for nodes in nodes if nodes is not None}
        self.connections = [c for c in connections]
        self.fitness = 0

    def _path_exist(self, start_node_id, end_node_id, checked_nodes=None):
        if checked_nodes is None:
            checked_nodes = set()

        if start_node_id == end_node_id:
            return True

        checked_nodes.add(start_node_id)
        for conn in self.connections:
            if conn.enabled and conn.in_node_id == start_node_id:
                if conn.out_node_id not in checked_nodes:
                    if self._path_exist(conn.out_node_id, end_node_id, checked_nodes):
                        return True
        return False

    def get_node(self, node_id):
        return self.nodes.get(node_id, None)

    def evaluate(self, inputs_values: list[float]):
        node_values = {}
        node_inputs = {n.id: [] for n in self.nodes.values()}

        input_nodes = [n for n in self.nodes.values() if n.layer == "input"]
        output_nodes = [n for n in self.nodes.values() if n.layer == "output"]

        if len(inputs_values) != len(input_nodes):
            raise ValueError(
                "Le nombre de valeurs d'entrée ne correspond pas au nombre de noeuds d'entrée."
            )

        for node, val in zip(input_nodes, inputs_values):
            node_values[node.id] = val

        edges = {n.id: [] for n in self.nodes.values()}
        for conn in self.connections:
            if conn.enabled:
                edges[conn.in_node_id].append(conn.out_node_id)
                node_inputs[conn.out_node_id].append(conn)

        sorted_node_ids = topological_sort(edges)

        for node_id in sorted_node_ids:
            if node_id in node_values:
                continue

            incoming = node_inputs[node_id]
            node = self.nodes[node_id]

            if isinstance(node.aggregation, str):
                aggregation_func = aggregation_map.get(
                    node.aggregation, aggregation_map["sum"]
                )
            else:
                aggregation_func = node.aggregation

            if isinstance(node.activation, str):
                activation_func = activation_map.get(
                    node.activation, activation_map["sigmoid"]
                )
            else:
                activation_func = node.activation

            weighted_inputs = [
                conn.weight * node_values[conn.in_node_id] for conn in incoming
            ]

            if weighted_inputs:
                aggregated_value = aggregation_func(weighted_inputs)
            else:
                aggregated_value = 0.0

            total_input = aggregated_value + node.biais
            node_values[node_id] = activation_func(total_input)

        return [node_values.get(node.id, 0) for node in output_nodes]

    def check_connection(self, node1, node2):
        for conn in self.connections:
            if (conn.in_node_id == node1.id and conn.out_node_id == node2.id) or (
                conn.in_node_id == node2.id and conn.out_node_id == node1.id
            ):
                return True
        return False

    def mutate_add_connection(self, innov: InovationsTracker):
        node_list = list(self.nodes.values())

        if len(node_list) < 2:
            return

        max_tries = 20

        for _ in range(max_tries):
            node1 = random.choice(
                [node for node in node_list if node.layer != "output"]
            )
            node2 = random.choice([node for node in node_list if node.layer != "input"])

            if node1.id == node2.id:
                continue

            if node1.layer == "output" or node2.layer == "input":
                continue

            if self.check_connection(node1, node2):
                continue

            # Vérifier les cycles
            if self._path_exist(node2.id, node1.id):
                continue

            # Créer la connexion
            innov_num = innov.get_connection_innovation(node1.id, node2.id)
            new_conn = ConnectionGene(
                in_node_id=node1.id,
                out_node_id=node2.id,
                weight=random.uniform(-1, 1),
                innov=innov_num,
                enabled=True,
            )
            self.connections.append(new_conn)
            return

    def mutate_add_node(self, innov: InovationsTracker):
        enabled_conn = [c for c in self.connections if c.enabled]
        if not enabled_conn:
            return
        connection = random.choice(enabled_conn)
        connection.enabled = False

        node_id, conn1_innov, conn2_innov = innov.get_node_innovation(connection.innov)

        activation_choice = random.choice(["sigmoid", "tanh"])
        aggregation_choice = random.choice(["sum", "max", "min", "mul", "avg"])

        new_node = NodeGene(
            node_id,
            "hidden",
            activation_choice,
            random.uniform(-1, 1),
            aggregation_choice,
        )

        conn1 = ConnectionGene(
            in_node_id=connection.in_node_id,
            out_node_id=node_id,
            weight=1.0,
            innov=conn1_innov,
            enabled=True,
        )
        conn2 = ConnectionGene(
            in_node_id=node_id,
            out_node_id=connection.out_node_id,
            weight=connection.weight,
            innov=conn2_innov,
            enabled=True,
        )

        self.nodes[node_id] = new_node
        self.connections.extend([conn1, conn2])

    def mutate_weights(self, rate=0.8, power=0.5):
        """Muter les poids des connexions"""
        for conn in self.connections:
            if random.random() < rate:
                if random.random() < 0.1:
                    # Remplacement complet
                    conn.weight = random.uniform(-2, 2)
                else:
                    # Perturbation
                    conn.weight += random.gauss(0, power)
                    conn.weight = max(-5, min(5, conn.weight))

    def mutate_bias(self, rate=0.7, power=0.5):
        """Muter les biais des nœuds"""
        for node in self.nodes.values():
            if node.layer != "input" and random.random() < rate:
                if random.random() < 0.1:
                    node.biais = random.uniform(-2, 2)
                else:
                    node.biais += random.gauss(0, power)
                    node.biais = max(-5, min(5, node.biais))

    def mutate_activation(self, rate=0.05):
        """Muter l'activation des nœuds cachés et de sortie"""
        activations = ["sigmoid", "tanh", "relu", "sin", "gauss"]
        for node in self.nodes.values():
            if node.layer != "input" and random.random() < rate:
                old_activation = node.activation
                new_activation = random.choice(activations)
                node.activation = new_activation

    def mutate_aggregation(self, rate=0.05):
        """AJOUTÉ: Muter l'agrégation des nœuds cachés et de sortie"""
        aggregations = ["sum", "max", "min", "mul", "avg", "maxabs"]
        for node in self.nodes.values():
            if node.layer != "input" and random.random() < rate:
                node.aggregation = random.choice(aggregations)

    def mutate_toggle_enable(self, rate=0.01):
        """AJOUTÉ: Désactiver/réactiver une connexion aléatoire"""
        if random.random() < rate and self.connections:
            conn = random.choice(self.connections)
            conn.enabled = not conn.enabled

    def mutate_delete_connection(self, rate=0.01):
        """AJOUTÉ: Supprimer une connexion (rarement utilisé)"""
        if random.random() < rate and len(self.connections) > 1:
            # Ne pas supprimer les connexions critiques
            non_critical = [
                c
                for c in self.connections
                if not c.enabled or len(self.connections) > 3
            ]
            if non_critical:
                conn_to_remove = random.choice(non_critical)
                self.connections.remove(conn_to_remove)

    def mutate(
        self,
        innov,
        weight_mutation_rate=0.8,
        bias_mutation_rate=0.7,
        conn_mutation_rate=0.1,
        node_mutation_rate=0.05,
        activation_mutation_rate=0.05,
        aggregation_mutation_rate=0.05,
        toggle_enable_rate=0.01,
        delete_conn_rate=0.005,
    ):
        """
        Applique toutes les mutations possibles

        Taux recommandés pour XOR:
        - Poids: 0.8 (80% des connexions mutées)
        - Biais: 0.7 (70% des nœuds mutés)
        - Connexion: 0.1 (10% de chance d'ajouter)
        - Nœud: 0.05 (5% de chance d'ajouter)
        - Activation: 0.05 (5% de chance de changer)
        - Agrégation: 0.05 (5% de chance de changer)
        - Toggle: 0.01 (1% de chance de désactiver/activer)
        - Delete: 0.005 (0.5% de chance de supprimer)
        """

        # Mutations de base (toujours appliquées selon rate)
        self.mutate_weights(weight_mutation_rate)
        self.mutate_bias(bias_mutation_rate)

        # Mutations structurelles (probabilistes)
        if random.random() < conn_mutation_rate:
            self.mutate_add_connection(innov)

        if random.random() < node_mutation_rate:
            self.mutate_add_node(innov)

        # Mutations de propriétés (probabilistes)
        if random.random() < activation_mutation_rate:
            self.mutate_activation()

        if random.random() < aggregation_mutation_rate:
            self.mutate_aggregation()

        # Mutations topologiques rares
        if random.random() < toggle_enable_rate:
            self.mutate_toggle_enable()

        if random.random() < delete_conn_rate:
            self.mutate_delete_connection()

    def copy(self):
        # Copier les nœuds
        new_nodes = []
        for node in self.nodes.values():
            new_node = NodeGene(
                id=node.id,
                layer=node.layer,
                activation=node.activation,
                biais=node.biais,
                aggregation=node.aggregation,
            )
            new_nodes.append(new_node)

        # Copier les connexions
        new_connections = []
        for conn in self.connections:
            new_conn = ConnectionGene(
                in_node_id=conn.in_node_id,
                out_node_id=conn.out_node_id,
                weight=conn.weight,
                innov=conn.innov,
                enabled=conn.enabled,
            )
            new_connections.append(new_conn)

        new_genome = Genome(new_nodes, new_connections)
        new_genome.fitness = self.fitness
        return new_genome

    def __str__(self):
        return f"Genome(nodes={len(self.nodes)}, connections={len(self.connections)}, fitness={self.fitness:.3f})"

    def __repr__(self):
        return self.__str__()
