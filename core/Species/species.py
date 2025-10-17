from core.Genome.genome import Genome
class Species:
    def __init__(self, representative:Genome):
        self.representative = representative
        self.members = [representative]
        self.adjusted_fitness = 0
        self.best_fitness = -float('inf')
        self.stagnante_gen = 0
        
    
    
    def add_members(self, member:Genome):
        self.members.append(member)
    
    def clear_members(self):
        self.members = []
    
    
    def update_fitness(self):
        if not self.members:
            self.adjusted_fitness = 0
            return

        current_best_fitness = max(member.fitness for member in self.members)
        
        if current_best_fitness > self.best_fitness:
            self.best_fitness = current_best_fitness
            self.stagnante_gen = 0
        else:
            self.stagnante_gen += 1
        
        self.adjusted_fitness = sum(member.fitness for member in self.members) / len(self.members)