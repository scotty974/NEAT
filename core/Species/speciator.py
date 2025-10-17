from core.Genome.genome import Genome
from utils.distance import distance
from core.Species.species import Species
class Speciator:
    def __init__(self, compatibility_threshold=3.0):
        self.compatibility_threshold = compatibility_threshold
        self.species = []
        
    #  fonction de speciation permettant de diviser la population en especes
    def speciate(self, population:list[Genome]):
        # vider la liste des especes
        for s in self.species:
            s.clear_members()
        
        # ajouter les genomes aux especes
        for genome in population:
            found_species = False
            for species in self.species:
                if distance(genome, species.representative) < self.compatibility_threshold:
                    species.add_members(genome)
                    found_species = True
                    break
            if not found_species:
                new_species = Species(representative=genome)
                self.species.append(new_species)
        
        # supprimer les especes sans membres
        self.species = [s for s in self.species if s.members]

        # mise à jour des especes
        for species in self.species:
            species.update_fitness()
            
            species.representative = max(species.members, key=lambda g: g.fitness)
            
    def get_species(self):
        return self.species