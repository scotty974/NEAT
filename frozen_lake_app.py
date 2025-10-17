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
    impossible_actions = 0
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"Starting new episode - Initial position: {observation}")
        print(f"Goal position: {goal_position}")
        print(f"{'='*50}")
    
    while not done and steps < max_steps:
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
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        
        # Détecter action impossible (pas de mouvement)
        if old_observation == observation and not done:
            impossible_actions += 1
            if verbose:
                print(f"  ⚠️ Impossible action! (tried to move outside map)")
        
        # Détecter si on a atteint le goal
        if reward > 0:
            reached_goal = True
        
        if verbose:
            print(f"  New position: {observation} (row {observation//MAP_SIZE}, col {observation%MAP_SIZE})")
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
        print(f"Impossible actions: {impossible_actions}")
        print(f"{'='*50}\n")
    
    return reached_goal, steps, impossible_actions

def calculate_fitness(reached_goal, steps, impossible_actions, optimal_steps=6):
    """
    Calcule la fitness sur 100 points.
    
    Args:
        reached_goal: True si le goal a été atteint
        steps: Nombre de pas effectués
        impossible_actions: Nombre d'actions impossibles tentées
        optimal_steps: Nombre optimal de pas pour résoudre (6 pour 4x4)
    
    Returns:
        fitness: Score entre 0 et 100
    """
    if not reached_goal:
        # Si échec: score faible avec pénalité pour actions impossibles
        fitness = max(0, 10 - (impossible_actions * 2))
        return fitness
    
    # Si succès: score de base de 50 points
    fitness = 50
    
    # Bonus pour chemin court (max 40 points)
    # Plus on est proche de l'optimal, plus le bonus est élevé
    if steps <= optimal_steps:
        # Chemin optimal ou mieux: bonus complet
        path_bonus = 40
    else:
        # Bonus décroissant en fonction de l'écart avec l'optimal
        # On tolère jusqu'à 3x l'optimal avant d'avoir 0 bonus
        max_acceptable_steps = optimal_steps * 3
        if steps <= max_acceptable_steps:
            path_bonus = 40 * (1 - (steps - optimal_steps) / (max_acceptable_steps - optimal_steps))
        else:
            path_bonus = 0
    
    fitness += path_bonus
    
    # Pénalité pour actions impossibles (max -10 points)
    impossible_penalty = min(10, impossible_actions * 2)
    fitness -= impossible_penalty
    
    # S'assurer que la fitness reste entre 0 et 100
    fitness = max(0, min(100, fitness))
    
    return fitness

def eval_genome(genome, num_episodes=50):
    """Évalue un genome sur plusieurs épisodes."""
    env = gym.make('FrozenLake-v1', is_slippery=True)
    
    episode_fitnesses = []
    
    for _ in range(num_episodes):
        try:
            reached_goal, steps, impossible_actions = run_episode(genome, env)
            fitness = calculate_fitness(reached_goal, steps, impossible_actions)
            episode_fitnesses.append(fitness)
        except Exception as e:
            # En cas d'erreur, fitness de 0
            episode_fitnesses.append(0)
    
    env.close()
    
    # Fitness moyenne sur tous les épisodes
    avg_fitness = sum(episode_fitnesses) / len(episode_fitnesses)
    
    return avg_fitness

def watch_genome_play(genome_path='best_genome_frozenlake.pkl', num_episodes=5, render_mode='human'):
    """
    Charge et visualise un genome sauvegardé en train de jouer.
    
    Args:
        genome_path: Chemin vers le fichier pickle du genome
        num_episodes: Nombre d'épisodes à visualiser
        render_mode: 'human' pour affichage graphique, 'ansi' pour affichage texte
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
        
        reached_goal, steps, impossible_actions = run_episode(genome, env, render=True, verbose=True)
        fitness = calculate_fitness(reached_goal, steps, impossible_actions)
        
        if reached_goal:
            successes += 1
            print(f"✓ Episode {ep + 1}: SUCCESS in {steps} steps! (Fitness: {fitness:.1f}/100)")
        else:
            print(f"✗ Episode {ep + 1}: Failed after {steps} steps (Fitness: {fitness:.1f}/100)")
        
        total_steps_list.append(steps)
        
        if ep < num_episodes - 1:
            import time
            time.sleep(1)  # Pause entre les épisodes
    
    env.close()
    
    # Statistiques finales
    print(f"\n{'='*70}")
    print(f"{'FINAL STATISTICS':^70}")
    print(f"{'='*70}")
    print(f"Success Rate: {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")
    print(f"Average Steps: {sum(total_steps_list)/len(total_steps_list):.1f}")
    if successes > 0:
        successful_steps = [s for i, s in enumerate(total_steps_list) if i < len(total_steps_list)]
        print(f"Steps range: {min(total_steps_list)} - {max(total_steps_list)}")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    import sys
    
    # Mode de lancement
    if len(sys.argv) > 1:
        if sys.argv[1] == 'watch':
            # Mode visualisation
            num_eps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            genome_file = sys.argv[3] if len(sys.argv) > 3 else 'best_genome_frozenlake.pkl'
            watch_genome_play(genome_file, num_eps)
            sys.exit(0)
        elif sys.argv[1] == 'test':
            # Mode test rapide (sans affichage graphique)
            genome_file = sys.argv[2] if len(sys.argv) > 2 else 'best_genome_frozenlake.pkl'
            
            print("\nLoading genome for quick test...")
            with open(genome_file, 'rb') as f:
                genome = pickle.load(f)
            
            env = gym.make('FrozenLake-v1', is_slippery=True)
            successes = 0
            total_fitness = 0
            for i in range(100):
                reached_goal, steps, impossible_actions = run_episode(genome, env)
                fitness = calculate_fitness(reached_goal, steps, impossible_actions)
                total_fitness += fitness
                if reached_goal:
                    successes += 1
            env.close()
            
            avg_fitness = total_fitness / 100
            print(f"Test on 100 episodes:")
            print(f"  Success rate: {successes}/100 ({100*successes/100:.1f}%)")
            print(f"  Average fitness: {avg_fitness:.1f}/100")
            sys.exit(0)
    
    # Mode entraînement (par défaut)
    generations = 200
    pop_size = 150
    target_fitness = 80  # Sur 100
    speciator_threshold = 1.0
    save_best = True
    NUM_INPUTS = 16  # 16 états possibles (4x4 grid) en one-hot
    NUM_OUTPUTS = 4  # 4 actions (gauche, bas, droite, haut)
    
    innov = InovationsTracker()
    speciator = Speciator(speciator_threshold)
    population = Manager(NUM_INPUTS, NUM_OUTPUTS, innov).create_init_pop(pop_size)
    
    best_fitness_history = []
    avg_fitness_history = []
    species_count_history = []
    
    print("=" * 100)
    print(f"{'NEAT - FrozenLake Problem (From Scratch)':^100}")
    print("=" * 100)
    print(f"Population: {pop_size} | Target: {target_fitness}/100 | Speciation Threshold: {speciator_threshold}")
    print(f"Inputs: {NUM_INPUTS} (one-hot encoded states) | Outputs: {NUM_OUTPUTS} (actions)")
    print("\nFitness System (out of 100):")
    print("  - Success: 50 base points")
    print("  - Short path bonus: up to 40 points (optimal = 6 steps)")
    print("  - Impossible actions: -2 points each")
    print("  - Failure: max 10 points (with penalties)")
    print("=" * 100)
    
    stagnation_counter = 0
    best_ever = 0
    
    for gen in range(generations):
        fitness_score = []
        
        print(f"Evaluating generation {gen}...", end=" ", flush=True)
        
        for i, genome in enumerate(population):
            if i % 30 == 0:
                print(".", end="", flush=True)
            
            score = eval_genome(genome, num_episodes=50)
            fitness_score.append(score)
        
        print(" Done!")
        
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
        if gen % 10 == 0 or best_fitness >= target_fitness or gen < 5:
            print(f"\n{'─' * 100}")
            print(f"Gen {gen:04} | Best: {best_fitness:.1f}/100 | Avg: {avg_fitness:.1f}/100 | "
                  f"Species: {num_species} | Stagnation: {stagnation_counter}")
            print(f"{'─' * 100}")
            
            # Test du meilleur génome
            best_idx = fitness_score.index(best_fitness)
            best_genome = population[best_idx]
            
            print(f"\nTesting best genome (visual test on 5 episodes):")
            env = gym.make('FrozenLake-v1', is_slippery=True)
            test_fitnesses = []
            test_details = []
            for ep in range(5):
                reached_goal, steps, impossible_actions = run_episode(best_genome, env)
                fitness = calculate_fitness(reached_goal, steps, impossible_actions)
                test_fitnesses.append(fitness)
                test_details.append((reached_goal, steps, impossible_actions))
                
                result = "✓ SUCCESS" if reached_goal else "✗ Failed"
                print(f"  Ep {ep+1}: {result} | Steps: {steps:2d} | Impossible: {impossible_actions} | Fitness: {fitness:.1f}/100")
            
            env.close()
            avg_test_fitness = sum(test_fitnesses) / len(test_fitnesses)
            successes = sum(1 for r, _, _ in test_details if r)
            print(f"  Average fitness: {avg_test_fitness:.1f}/100 | Success rate: {successes}/5")
            
            print(f"\nSpecies breakdown:")
            for i, species in enumerate(species_list, 1):
                members_count = len(species.members)
                try:
                    # Récupérer les fitness des membres de cette espèce
                    species_fitnesses = []
                    for member in species.members:
                        try:
                            idx = population.index(member)
                            species_fitnesses.append(fitness_score[idx])
                        except (ValueError, IndexError) as e:
                            # Le membre n'est pas dans la population (ne devrait pas arriver)
                            continue
                    
                    if species_fitnesses:
                        species_avg = sum(species_fitnesses) / len(species_fitnesses)
                        species_best = max(species_fitnesses)
                    else:
                        species_avg = 0
                        species_best = 0
                    
                    bar_length = int(members_count / pop_size * 30)
                    bar = "█" * bar_length + "░" * (30 - bar_length)
                    
                    print(f"  S{i:02} | {bar} | N={members_count:3d} ({members_count/pop_size*100:5.1f}%) | "
                          f"Avg: {species_avg:.1f} | Best: {species_best:.1f}")
                except Exception as e:
                    # Afficher l'erreur pour débugger
                    print(f"  S{i:02} | Error calculating stats: {e}")
                    print(f"         Members: {members_count}, Population size: {len(population)}")
            
            # Warning si une seule espèce
            if num_species == 1:
                print(f"\n⚠️  WARNING: Only 1 species! Diversity lost.")
        else:
            print(f"Gen {gen:04} | Best: {best_fitness:.1f}/100 | Avg: {avg_fitness:.1f}/100 | "
                  f"Species: {num_species:2d} | Stagnation: {stagnation_counter:3d}", end="")
            if num_species == 1:
                print(" ⚠️", end="")
            print()
        
        if best_fitness >= target_fitness:
            print(f"\n{'=' * 100}")
            print(f"{'🎉 TARGET REACHED! 🎉':^100}")
            print(f"{'=' * 100}")
            print(f"Solution found in generation {gen}")
            print(f"Best fitness: {best_fitness:.1f}/100")
            
            best_genome = population[fitness_score.index(best_fitness)]
            
            if save_best:
                with open("best_genome_frozenlake.pkl", "wb") as f:
                    pickle.dump(best_genome, f)
                print("\n✓ Best genome saved to 'best_genome_frozenlake.pkl'")
            break
        
        # Warning si stagnation prolongée
        if stagnation_counter > 50 and stagnation_counter % 25 == 0:
            print(f"\n⚠️  STAGNATION: No improvement for {stagnation_counter} generations")
        
        population = Manager.evolution(population, fitness_score, speciator, innov)
    
    print(f"\n{'=' * 100}")
    print(f"{'TRAINING COMPLETE':^100}")
    print(f"{'=' * 100}")
    print(f"Final Best Fitness: {best_fitness:.1f}/100")
    print(f"Final Avg Fitness: {avg_fitness:.1f}/100")
    print(f"Target Fitness: {target_fitness}/100")
    print(f"Generations: {gen + 1} / {generations}")
    print(f"Final Species: {len(speciator.get_species())}")
    print(f"Max Species Seen: {max(species_count_history)}")
    print(f"Best Ever: {best_ever:.1f}/100")
    
    if best_fitness < target_fitness:
        print(f"\n⚠️  Target not reached. Best fitness: {best_fitness:.1f}/100 < {target_fitness}/100")
    
    print("=" * 100)
    
    # Test final approfondi du meilleur génome
    print("\n" + "="*50)
    print("FINAL TEST - 100 episodes")
    print("="*50)
    
    best_genome = population[fitness_score.index(best_fitness)]
    env = gym.make('FrozenLake-v1', is_slippery=True)
    
    final_successes = 0
    final_fitnesses = []
    final_steps = []
    final_impossible = []
    
    for _ in range(100):
        reached_goal, steps, impossible_actions = run_episode(best_genome, env)
        fitness = calculate_fitness(reached_goal, steps, impossible_actions)
        final_fitnesses.append(fitness)
        
        if reached_goal:
            final_successes += 1
            final_steps.append(steps)
        final_impossible.append(impossible_actions)
    
    env.close()
    
    avg_fitness = sum(final_fitnesses) / len(final_fitnesses)
    avg_impossible = sum(final_impossible) / len(final_impossible)
    
    print(f"Success Rate: {final_successes}/100 ({100*final_successes/100:.1f}%)")
    print(f"Average Fitness: {avg_fitness:.1f}/100")
    print(f"Average Impossible Actions: {avg_impossible:.2f}")
    if final_steps:
        avg_steps = sum(final_steps) / len(final_steps)
        print(f"Average steps (successful episodes): {avg_steps:.1f}")
        print(f"Steps range: {min(final_steps)} - {max(final_steps)}")
        print(f"Optimal steps: 6")
    
    # Visualiser le meilleur genome
    try:
        plot_genome(best_genome)
    except:
        print("\nNote: Genome visualization not available")