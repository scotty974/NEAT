from core.tracker.trackerInnovations import InovationsTracker
from core.Node.node import NodeGene, ConnectionGene
import random
from core.Genome.genome import Genome
from core.Species.speciator import Speciator
from utils.crossover import crossover


class Manager:
    def __init__(self, nb_inputs, nb_outputs, innov: InovationsTracker):
        self.innov = innov
        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs

        self.inputs_node = []
        self.outputs_node = []
        self.connections = []
        self.activations = [
            "sigmoid",
            "relu",
            "tanh",
            "sin",
            "gauss",
            "identity",
            "clamped",
            "inv",
            "log",
            "exp",
            "abs",
            "hat",
            "square",
            "cube",
            "softplus",
        ]
        self.aggregations = [
            "sum",
            "max",
            "min",
            "mul",
            "avg",
            "median",
            "maxabs",
            "product",
        ]

    def create_genome_initial(self):
        for i in range(self.nb_inputs):
            node = NodeGene(
                i,
                "input",
                random.choice(self.activations),
                random.uniform(-1, 1),
                random.choice(self.aggregations),
            )
            self.inputs_node.append(node)

        for i in range(self.nb_outputs):
            node_id = self.nb_inputs + i
            node = NodeGene(
                node_id,
                "output",
                random.choice(self.activations),
                random.uniform(-1, 1),
                random.choice(self.aggregations),
            )
            self.outputs_node.append(node)

        self.innov.node_id_counter = self.nb_inputs + self.nb_outputs

        for i in range(self.nb_inputs):
            for j in range(self.nb_outputs):
                in_node_id = i
                out_node_id = self.nb_inputs + j
                innov_num = self.innov.get_connection_innovation(
                    in_node_id, out_node_id
                )
                weight = random.uniform(-1, 1)
                conn = ConnectionGene(
                    innov_num, in_node_id, out_node_id, weight, innov_num
                )
                self.connections.append(conn)

        all_nodes = self.inputs_node + self.outputs_node
        genomes = Genome(all_nodes, self.connections)
        return genomes

    def create_init_pop(self, pop_size):
        population = []
        for _ in range(pop_size):
            population.append(self.create_genome_initial())
        return population

    @staticmethod
    def reproduce_species(species, offspring_count, innov: InovationsTracker):
        offspring = []
        members = species.members

        if not members:
            return offspring

        for _ in range(offspring_count):
            parent1 = random.choice(members)

            if random.random() < 0.75 and len(members) > 1:  # crossover 75%
                parent2 = random.choice(members)
                # Crossover crée déjà une copie
                child = crossover(parent1, parent2)
            else:
                child = parent1.copy()

            # Mutations avec taux améliorés
            child.mutate_weights(rate=0.8, power=0.5)
            child.mutate_bias(rate=0.7, power=0.5)

            # Mutations structurelles
            if random.random() < 0.1:
                child.mutate_add_connection(innov)

            if random.random() < 0.05:
                child.mutate_add_node(innov)

            # Mutations de propriétés
            if random.random() < 0.05:
                child.mutate_activation()

            if random.random() < 0.05:  # AJOUTÉ : mutation d'agrégation
                child.mutate_aggregation()

            # Mutation toggle (rare)
            if random.random() < 0.01:
                child.mutate_toggle_enable()

            offspring.append(child)

        return offspring

    def evolution(
        population,
        fitness_scores,
        speciator: Speciator,
        innov: InovationsTracker,
        stagnation_limit: int = 15,
    ):
        new_population = []

        for genome, fitness in zip(population, fitness_scores):
            genome.fitness = fitness

        speciator.speciate(population)
        species_list = speciator.get_species()
        species_list.sort(key=lambda s: s.best_fitness, reverse=True)

        surviving_species = []
        if species_list:
            surviving_species.append(species_list[0])
        for s in species_list[1:]:
            if s.stagnante_gen < stagnation_limit:
                surviving_species.append(s)

        species_list = surviving_species

        total_adjusted_fitness = sum(s.adjusted_fitness for s in species_list)

        for species in species_list:
            if species.members:
                best_genome = sorted(
                    species.members, key=lambda g: g.fitness, reverse=True
                )
                n_elite = max(1, int(len(best_genome) * 0.2))
                new_population.extend(best_genome[:n_elite])

        remaining_offspring = len(population) - len(new_population)

        for species in species_list:
            if total_adjusted_fitness > 0:
                offspring_count = int(
                    (species.adjusted_fitness / total_adjusted_fitness)
                    * remaining_offspring
                )
            else:
                offspring_count = remaining_offspring // len(species_list)
            if offspring_count > 0:
                offspring = Manager.reproduce_species(species, offspring_count, innov)
                new_population.extend(offspring)

        while len(new_population) < len(population):
            best_species = max(species_list, key=lambda s: s.adjusted_fitness)
            offspring = Manager.reproduce_species(best_species, 1, innov)
            new_population.extend(offspring)

        return new_population
