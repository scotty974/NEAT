from core.tracker.trackerInnovations import InovationsTracker
from core.Species.speciator import Speciator
from core.managers.manager import Manager
import pickle
from utils.vis import plot_genome

if __name__ == '__main__':
    generations = 1000
    pop_size = 150  # Augmenté pour plus de diversité
    target_fitness = 3.95  # Seuil légèrement plus réaliste
    speciator_threshold = 3.0  # RÉDUIT pour favoriser plus d'espèces
    save_best = True
    NUM_INPUTS = 2
    NUM_OUTPUTS = 1
    
    innov = InovationsTracker()
    speciator = Speciator(speciator_threshold)
    population = Manager(NUM_INPUTS, NUM_OUTPUTS, innov).create_init_pop(pop_size)
    
    best_fitness_history = []
    avg_fitness_history = []
    species_count_history = []
    
    X = [[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [0, 1], [1, 0], [1, 1]]
    Y = [0, 1, 1, 0]
    
    print("=" * 100)
    print(f"{'NEAT - XOR Problem (IMPROVED)':^100}")
    print("=" * 100)
    print(f"Population: {pop_size} | Target: {target_fitness} | Speciation Threshold: {speciator_threshold}")
    print("=" * 100)
    
    stagnation_counter = 0
    best_ever = 0
    
    for gen in range(generations):
        fitness_score = []
        outputs_debug = []
        
        for genome in population:
            score = 0
            genome_outputs = []
            
            for x, y in zip(X, Y):
                try:
                    output = genome.evaluate(x)
                    if output:
                        error = abs(output[0] - y)
                        score += max(0, 1 - error)
                        genome_outputs.append(output[0])
                    else:
                        score += 0
                        genome_outputs.append(None)
                except Exception as e:
                    score += 0
                    genome_outputs.append(None)
            
            # Bonus pour résolution parfaite
            if score >= 3.95:
                score += 0.5
            
            fitness_score.append(score)
            outputs_debug.append(genome_outputs)
        
        best_fitness = max(fitness_score)
        avg_fitness = sum(fitness_score) / len(fitness_score)
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        
        species_list = speciator.get_species()
        num_species = len(species_list)
        species_count_history.append(num_species)
        
        # Détection de stagnation
        if best_fitness > best_ever:
            best_ever = best_fitness
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        
        # Affichage détaillé
        if gen % 50 == 0 or best_fitness >= target_fitness or gen < 5:
            print(f"\n{'─' * 100}")
            print(f"Gen {gen:04} | Best: {best_fitness:.4f} | Avg: {avg_fitness:.4f} | "
                  f"Species: {num_species} | Stagnation: {stagnation_counter}")
            print(f"{'─' * 100}")
            
            # Afficher les outputs du meilleur génome
            best_idx = fitness_score.index(best_fitness)
            best_outputs = outputs_debug[best_idx]
            print(f"Best genome outputs:")
            for i, (x, y, out) in enumerate(zip(X, Y, best_outputs)):
                if out is not None:
                    print(f"  {x} -> {out:.4f} (expected {y}, error: {abs(out-y):.4f})")
                else:
                    print(f"  {x} -> ERROR (expected {y})")
            
            print(f"\nSpecies breakdown:")
            for i, species in enumerate(species_list, 1):
                members_count = len(species.members)
                try:
                    species_fitnesses = [fitness_score[population.index(m)] for m in species.members if m in population]
                    species_avg = sum(species_fitnesses) / len(species_fitnesses) if species_fitnesses else 0
                    species_best = max(species_fitnesses) if species_fitnesses else 0
                    
                    bar_length = int(members_count / pop_size * 40)
                    bar = "█" * bar_length + "░" * (40 - bar_length)
                    
                    print(f"  S{i:02} | {bar} | N={members_count:3d} ({members_count/pop_size*100:5.1f}%) | "
                          f"Avg: {species_avg:.3f} | Best: {species_best:.3f}")
                except:
                    pass
            
            # Warning si une seule espèce
            if num_species == 1:
                print(f"\n⚠️  WARNING: Only 1 species! Diversity lost. Consider:")
                print(f"   - Decreasing speciation threshold (currently {speciator_threshold})")
                print(f"   - Increasing mutation rates")
                print(f"   - Checking compatibility distance calculation")
        else:
            print(f"Gen {gen:04} | Best: {best_fitness:.4f} | Avg: {avg_fitness:.4f} | "
                  f"Species: {num_species:2d} | Stagnation: {stagnation_counter:3d}", end="")
            if num_species == 1:
                print(" ⚠️", end="")
            print()
        
        if best_fitness >= target_fitness:
            print(f"\n{'=' * 100}")
            print(f"{'🎉 PROBLEM SOLVED! 🎉':^100}")
            print(f"{'=' * 100}")
            print(f"Solution found in generation {gen}")
            print(f"Best fitness: {best_fitness:.4f}")
            
            best_genome = population[fitness_score.index(best_fitness)]
            print(f"\nFinal outputs:")
            for x, y in zip(X, Y):
                output = best_genome.evaluate(x)
                if output:
                    print(f"  {x} -> {output[0]:.4f} (expected {y})")
            
            if save_best:
                with open("best_genome.pkl", "wb") as f:
                    pickle.dump(best_genome, f)
                print("\n✓ Best genome saved to 'best_genome.pkl'")
            break
        
        # Warning si stagnation prolongée
        if stagnation_counter > 100 and stagnation_counter % 50 == 0:
            print(f"\n⚠️  STAGNATION DETECTED: No improvement for {stagnation_counter} generations")
        
        population = Manager.evolution(population, fitness_score, speciator, innov)
    
    print(f"\n{'=' * 100}")
    print(f"{'TRAINING COMPLETE':^100}")
    print(f"{'=' * 100}")
    print(f"Final Best Fitness: {best_fitness:.4f} / {target_fitness}")
    print(f"Final Avg Fitness: {avg_fitness:.4f}")
    print(f"Generations: {gen + 1} / {generations}")
    print(f"Final Species: {len(speciator.get_species())}")
    print(f"Max Species Seen: {max(species_count_history)}")
    print(f"Best Ever: {best_ever:.4f}")
    
    if best_fitness < target_fitness:
        print(f"\n⚠️  WARNING: Solution not found, best fitness: {best_fitness:.4f} < {target_fitness}")
    
    print("=" * 100)
    
    best_genome = population[fitness_score.index(best_fitness)]
    plot_genome(best_genome)