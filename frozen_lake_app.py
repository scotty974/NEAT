from core.tracker.trackerInnovations import InovationsTracker
from core.Species.speciator import Speciator
from core.managers.manager import Manager
from utils.vis import plot_genome
import gymnasium as gym
import numpy as np
import pickle

MAP_SIZE = 4  # Pour FrozenLake-v1 standard (4x4)

def manhattan_distance(pos1, pos2, map_size=MAP_SIZE):
    """Calcule la distance de Manhattan entre deux positions."""
    row1, col1 = pos1 // map_size, pos1 % map_size
    row2, col2 = pos2 // map_size, pos2 % map_size
    return abs(row1 - row2) + abs(col1 - col2)

def run_episode(genome, env, max_steps=100, render=False, verbose=False):
    """Exécute un épisode complet et retourne les informations nécessaires."""
    observation, info = env.reset()
    done = False
    steps = 0
    
    action_names = ['←', '↓', '→', '↑']
    
    # Position du goal (coin inférieur droit pour FrozenLake 4x4)
    goal_position = MAP_SIZE * MAP_SIZE - 1  # 15 pour 4x4
    
    reached_goal = False
    min_distance_to_goal = MAP_SIZE * 2  # Distance maximale possible
    total_distance_progress = 0
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"Starting new episode - Initial position: {observation}")
        print(f"Goal position: {goal_position}")
        print(f"{'='*50}")
    
    while not done and steps < max_steps:
        # Calculer distance actuelle au goal
        current_distance = manhattan_distance(observation, goal_position)
        min_distance_to_goal = min(min_distance_to_goal, current_distance)
        
        # Convertir l'observation en one-hot encoding
        state_input = np.zeros(MAP_SIZE * MAP_SIZE)
        state_input[observation] = 1.0
        
        # Obtenir l'action du réseau de neurones
        try:
            output = genome.evaluate(state_input.tolist())
            if output and len(output) >= 4:
                action = np.argmax(output[:4])
                if verbose:
                    print(f"\nStep {steps + 1}:")
                    print(f"  Position: {observation} (row {observation//MAP_SIZE}, col {observation%MAP_SIZE})")
                    print(f"  Distance to goal: {current_distance}")
                    print(f"  NN outputs: [{', '.join([f'{o:.3f}' for o in output[:4]])}]")
                    print(f"  Action: {action} ({action_names[action]})")
            else:
                action = 0
        except Exception as e:
            action = 0
            if verbose:
                print(f"  ERROR in NN evaluation: {e}")
        
        # Exécuter l'action
        old_observation = observation
        old_distance = current_distance
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        
        # Calculer le progrès vers le goal
        new_distance = manhattan_distance(observation, goal_position)
        distance_progress = old_distance - new_distance  # Positif si on se rapproche
        total_distance_progress += distance_progress
        
        # Détecter si on a atteint le goal
        if reward > 0:
            reached_goal = True
        
        if verbose:
            print(f"  New position: {observation} (row {observation//MAP_SIZE}, col {observation%MAP_SIZE})")
            print(f"  Distance progress: {distance_progress:+d} (now at {new_distance})")
            if reward > 0:
                print(f"  🎉 GOAL REACHED!")
            elif done:
                print(f"  ❌ Fell in hole!")
        
        if render:
            import time
            time.sleep(0.5)
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"Episode finished: {'SUCCESS ✓' if reached_goal else 'FAILED ✗'}")
        print(f"Total steps: {steps}")
        print(f"Min distance reached: {min_distance_to_goal}")
        print(f"Total progress: {total_distance_progress:+d}")
        print(f"{'='*50}\n")
    
    return reached_goal, steps, min_distance_to_goal, total_distance_progress

def calculate_fitness(reached_goal, steps, min_distance_to_goal, total_distance_progress, 
                     optimal_steps=6, max_distance=6):
    """
    Calcule la fitness sur 100 points, adaptée pour environnement stochastique.
    
    Args:
        reached_goal: True si le goal a été atteint
        steps: Nombre de pas effectués
        min_distance_to_goal: Distance minimale atteinte du goal
        total_distance_progress: Progrès cumulé vers le goal
        optimal_steps: Nombre optimal de pas (6 pour 4x4)
        max_distance: Distance maximale possible (6 pour 4x4)
    
    Returns:
        fitness: Score entre 0 et 100
    """
    if reached_goal:
        # SUCCÈS: Base de 60 points
        fitness = 60
        
        # Bonus pour chemin court (max 35 points)
        if steps <= optimal_steps:
            path_bonus = 35
        else:
            max_acceptable_steps = optimal_steps * 3
            if steps <= max_acceptable_steps:
                path_bonus = 35 * (1 - (steps - optimal_steps) / (max_acceptable_steps - optimal_steps))
            else:
                path_bonus = 0
        
        fitness += path_bonus
        
        # Bonus pour efficacité (max 5 points)
        # Récompense si peu d'étapes gaspillées
        efficiency = max(0, 1 - (steps - optimal_steps) / (max_acceptable_steps))
        fitness += efficiency * 5
        
    else:
        # ÉCHEC: Base de 0, mais récompenses pour progression
        fitness = 0
        
        # Récompense pour s'être approché du goal (max 40 points)
        # Plus on est proche, plus on gagne de points
        distance_score = (1 - min_distance_to_goal / max_distance) * 40
        fitness += distance_score
        
        # Bonus pour progression générale vers le goal (max 20 points)
        # Récompense le mouvement net vers le goal
        max_possible_progress = max_distance * 2  # Estimation généreuse
        progress_score = max(0, min(20, (total_distance_progress / max_possible_progress) * 20))
        fitness += progress_score
        
        # Petit bonus pour avoir survécu longtemps (max 10 points)
        survival_bonus = min(10, (steps / 100) * 10)
        fitness += survival_bonus
    
    # S'assurer que la fitness reste entre 0 et 100
    fitness = max(0, min(100, fitness))
    
    return fitness

def eval_genome(genome, num_episodes=100, verbose=False):
    """Évalue un genome sur plusieurs épisodes."""
    env = gym.make('FrozenLake-v1', is_slippery=True)
    
    episode_fitnesses = []
    successes = 0
    
    for ep in range(num_episodes):
        try:
            reached_goal, steps, min_dist, progress = run_episode(genome, env)
            fitness = calculate_fitness(reached_goal, steps, min_dist, progress)
            episode_fitnesses.append(fitness)
            if reached_goal:
                successes += 1
        except Exception as e:
            if verbose:
                print(f"Error in episode {ep}: {e}")
            episode_fitnesses.append(0)
    
    env.close()
    
    # Fitness moyenne sur tous les épisodes
    avg_fitness = sum(episode_fitnesses) / len(episode_fitnesses)
    
    # Bonus pour taux de succès (pour favoriser la consistance)
    success_rate = successes / num_episodes
    consistency_bonus = success_rate * 10  # Max 10 points
    
    final_fitness = min(100, avg_fitness + consistency_bonus)
    
    return final_fitness

def watch_genome_play(genome_path='best_genome_frozenlake.pkl', num_episodes=5, render_mode='human'):
    """
    Charge et visualise un genome sauvegardé en train de jouer.
    """
    print(f"\n{'='*70}")
    print(f"{'WATCHING GENOME PLAY':^70}")
    print(f"{'='*70}\n")
    
    try:
        with open(genome_path, 'rb') as f:
            genome = pickle.load(f)
        print(f"✓ Genome loaded from '{genome_path}'")
    except FileNotFoundError:
        print(f"✗ Error: File '{genome_path}' not found!")
        print("Please train a model first or provide the correct path.")
        return
    
    env = gym.make('FrozenLake-v1', is_slippery=True, render_mode=render_mode)
    
    print(f"\nPlaying {num_episodes} episodes with visualization...\n")
    
    successes = 0
    total_steps_list = []
    
    for ep in range(num_episodes):
        print(f"\n{'─'*70}")
        print(f"EPISODE {ep + 1}/{num_episodes}")
        print(f"{'─'*70}")
        
        reached_goal, steps, min_dist, progress = run_episode(genome, env, render=True, verbose=True)
        fitness = calculate_fitness(reached_goal, steps, min_dist, progress)
        
        if reached_goal:
            successes += 1
            print(f"✓ Episode {ep + 1}: SUCCESS in {steps} steps! (Fitness: {fitness:.1f}/100)")
        else:
            print(f"✗ Episode {ep + 1}: Failed after {steps} steps (Min dist: {min_dist}, Fitness: {fitness:.1f}/100)")
        
        total_steps_list.append(steps)
        
        if ep < num_episodes - 1:
            import time
            time.sleep(1)
    
    env.close()
    
    print(f"\n{'='*70}")
    print(f"{'FINAL STATISTICS':^70}")
    print(f"{'='*70}")
    print(f"Success Rate: {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")
    print(f"Average Steps: {sum(total_steps_list)/len(total_steps_list):.1f}")
    if successes > 0:
        successful_steps = [total_steps_list[i] for i in range(len(total_steps_list)) if i < successes]
        if successful_steps:
            print(f"Steps range (successes): {min(successful_steps)} - {max(successful_steps)}")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    import sys
    
    # Mode de lancement
    if len(sys.argv) > 1:
        if sys.argv[1] == 'watch':
            num_eps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            genome_file = sys.argv[3] if len(sys.argv) > 3 else 'best_genome_frozenlake.pkl'
            watch_genome_play(genome_file, num_eps)
            sys.exit(0)
        elif sys.argv[1] == 'test':
            genome_file = sys.argv[2] if len(sys.argv) > 2 else 'best_genome_frozenlake.pkl'
            
            print("\nLoading genome for quick test...")
            with open(genome_file, 'rb') as f:
                genome = pickle.load(f)
            
            env = gym.make('FrozenLake-v1', is_slippery=True)
            successes = 0
            total_fitness = 0
            for i in range(200):
                reached_goal, steps, min_dist, progress = run_episode(genome, env)
                fitness = calculate_fitness(reached_goal, steps, min_dist, progress)
                total_fitness += fitness
                if reached_goal:
                    successes += 1
            env.close()
            
            avg_fitness = total_fitness / 200
            print(f"Test on 200 episodes:")
            print(f"  Success rate: {successes}/200 ({100*successes/200:.1f}%)")
            print(f"  Average fitness: {avg_fitness:.1f}/100")
            sys.exit(0)
    
    # Mode entraînement
    generations = 300
    pop_size = 50
    target_fitness = 75  # Sur 100 (réduit car environnement stochastique)
    speciator_threshold = 1.0
    save_best = True
    NUM_INPUTS = 16
    NUM_OUTPUTS = 4
    
    innov = InovationsTracker()
    speciator = Speciator(speciator_threshold)
    population = Manager(NUM_INPUTS, NUM_OUTPUTS, innov).create_init_pop(pop_size)
    
    best_fitness_history = []
    avg_fitness_history = []
    species_count_history = []
    
    print("=" * 100)
    print(f"{'NEAT - FrozenLake Problem (Stochastic Environment)':^100}")
    print("=" * 100)
    print(f"Population: {pop_size} | Target: {target_fitness}/100 | Speciation: {speciator_threshold}")
    print(f"Environment: SLIPPERY (stochastic) - 33% direction, 66% random slide")
    print(f"Inputs: {NUM_INPUTS} (one-hot) | Outputs: {NUM_OUTPUTS} (actions)")
    print("\nFitness System (0-100, stochastic-aware):")
    print("  SUCCESS: 60 base + 35 path efficiency + 5 consistency = up to 100")
    print("  FAILURE: 0 base + 40 proximity + 20 progress + 10 survival = up to 70")
    print("  + Consistency bonus: up to 10 points based on success rate")
    print("=" * 100)
    
    stagnation_counter = 0
    best_ever = 0
    
    for gen in range(generations):
        fitness_score = []
        
        print(f"Evaluating generation {gen}...", end=" ", flush=True)
        
        for i, genome in enumerate(population):
            if i % 30 == 0:
                print(".", end="", flush=True)
            
            # Plus d'épisodes pour gérer la stochasticité
            score = eval_genome(genome, num_episodes=100)
            fitness_score.append(score)
        
        print(" Done!")
        
        best_fitness = max(fitness_score)
        avg_fitness = sum(fitness_score) / len(fitness_score)
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        
        species_list = speciator.get_species()
        num_species = len(species_list)
        species_count_history.append(num_species)
        
        if best_fitness > best_ever:
            best_ever = best_fitness
            stagnation_counter = 0
            
            # Sauvegarder le meilleur
            best_idx = fitness_score.index(best_fitness)
            best_genome = population[best_idx]
            with open("best_genome_frozenlake.pkl", "wb") as f:
                pickle.dump(best_genome, f)
        else:
            stagnation_counter += 1
        
        # Affichage détaillé
        if gen % 10 == 0 or best_fitness >= target_fitness or gen < 5:
            print(f"\n{'─' * 100}")
            print(f"Gen {gen:04} | Best: {best_fitness:.1f}/100 | Avg: {avg_fitness:.1f}/100 | "
                  f"Species: {num_species} | Stagnation: {stagnation_counter}")
            print(f"{'─' * 100}")
            
            # Test du meilleur génome
            best_idx = fitness_score.index(best_fitness)
            best_genome = population[best_idx]
            
            print(f"\nTesting best genome (10 episodes):")
            env = gym.make('FrozenLake-v1', is_slippery=True, render_mode=None)
            test_successes = 0
            test_fitnesses = []
            test_details = []
            
            for ep in range(10):
                reached_goal, steps, min_dist, progress = run_episode(best_genome, env)
                fitness = calculate_fitness(reached_goal, steps, min_dist, progress)
                test_fitnesses.append(fitness)
                test_details.append((reached_goal, steps, min_dist))
                
                if reached_goal:
                    test_successes += 1
                    result = f"✓ SUCCESS in {steps:2d} steps"
                else:
                    result = f"✗ Failed (min dist: {min_dist})"
                
                if ep < 5:  # Afficher seulement les 5 premiers
                    print(f"  Ep {ep+1}: {result} | Fitness: {fitness:.1f}")
            
            env.close()
            avg_test_fitness = sum(test_fitnesses) / len(test_fitnesses)
            print(f"  Success rate: {test_successes}/10 ({test_successes*10}%)")
            print(f"  Average fitness: {avg_test_fitness:.1f}/100")
            
            print(f"\nSpecies breakdown:")
            for i, species in enumerate(species_list, 1):
                members_count = len(species.members)
                species_fitnesses = []
                for member in species.members:
                    try:
                        idx = population.index(member)
                        species_fitnesses.append(fitness_score[idx])
                    except (ValueError, IndexError):
                        continue
                
                if species_fitnesses:
                    species_avg = sum(species_fitnesses) / len(species_fitnesses)
                    species_best = max(species_fitnesses)
                    bar_length = int(members_count / pop_size * 30)
                    bar = "█" * bar_length + "░" * (30 - bar_length)
                    print(f"  S{i:02} | {bar} | N={members_count:3d} | Avg: {species_avg:.1f} | Best: {species_best:.1f}")
            
            if num_species == 1:
                print(f"\n⚠️  WARNING: Only 1 species! Consider lowering speciation threshold.")
        else:
            print(f"Gen {gen:04} | Best: {best_fitness:.1f} | Avg: {avg_fitness:.1f} | "
                  f"Species: {num_species:2d} | Stag: {stagnation_counter:3d}", end="")
            if num_species == 1:
                print(" ⚠️", end="")
            print()
        
        if best_fitness >= target_fitness:
            print(f"\n{'=' * 100}")
            print(f"{'🎉 TARGET REACHED! 🎉':^100}")
            print(f"{'=' * 100}")
            break
        
        if stagnation_counter > 50 and stagnation_counter % 25 == 0:
            print(f"\n⚠️  STAGNATION: No improvement for {stagnation_counter} generations")
        
        population = Manager.evolution(population, fitness_score, speciator, innov)
    
    print(f"\n{'=' * 100}")
    print(f"{'TRAINING COMPLETE':^100}")
    print(f"{'=' * 100}")
    print(f"Final Best: {best_fitness:.1f}/100 | Target: {target_fitness}/100")
    print(f"Best Ever: {best_ever:.1f}/100")
    print(f"Generations: {gen + 1} / {generations}")
    print(f"Final Species: {len(speciator.get_species())}")
    print("=" * 100)
    
    # Test final
    print("\nFINAL TEST - 200 episodes")
    best_genome = population[fitness_score.index(best_fitness)]
    env = gym.make('FrozenLake-v1', is_slippery=True)
    
    final_successes = 0
    final_steps = []
    
    for _ in range(200):
        reached_goal, steps, _, _ = run_episode(best_genome, env)
        if reached_goal:
            final_successes += 1
            final_steps.append(steps)
    
    env.close()
    
    print(f"Success Rate: {final_successes}/200 ({100*final_successes/200:.1f}%)")
    if final_steps:
        print(f"Avg steps (successes): {sum(final_steps)/len(final_steps):.1f}")
        print(f"Steps range: {min(final_steps)} - {max(final_steps)}")
    
    try:
        plot_genome(best_genome)
    except:
        pass