import gymnasium as gym
import neat
import numpy as np
import pickle
from vis import plot_genome

# Configuration pour FrozenLake
MAP_SIZE = 4  # Pour FrozenLake-v1 standard (4x4)

def run_episode(net, env, render=False):
    """Exécute un épisode complet et retourne le score."""
    observation, info = env.reset()
    total_reward = 0
    done = False
    steps = 0
    max_steps = 100
    
    while not done and steps < max_steps:
        # Convertir l'observation en one-hot encoding
        state_input = np.zeros(MAP_SIZE * MAP_SIZE)
        state_input[observation] = 1.0
        
        # Obtenir l'action du réseau de neurones
        output = net.activate(state_input)
        action = np.argmax(output)
        
        # Exécuter l'action
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
    
    return total_reward, steps

def eval_genome(genome, config):
    """Évalue un genome sur plusieurs épisodes avec reward shaping."""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = gym.make('FrozenLake-v1', is_slippery=True)
    
    # Tester sur plusieurs épisodes
    num_episodes = 100
    total_rewards = 0
    successes = 0
    total_steps = 0
    successful_steps = []
    
    for _ in range(num_episodes):
        reward, steps = run_episode(net, env)
        total_rewards += reward
        total_steps += steps
        
        if reward > 0:
            successes += 1
            successful_steps.append(steps)
    
    env.close()
    
    # Reward shaping : bonus pour chemins efficaces
    efficiency_bonus = 0
    if successful_steps:
        # Récompenser les chemins courts (optimal = 6 pas pour 4x4)
        avg_steps = sum(successful_steps) / len(successful_steps)
        # Plus le chemin est court, plus le bonus est élevé
        efficiency_bonus = max(0, (100 - avg_steps) / 10)
    
    # Fitness = succès + bonus d'efficacité + pénalité pour chemins longs
    base_fitness = successes * 10  # 10 points par succès
    step_penalty = total_steps / num_episodes / 100  # Pénalité légère pour trop de pas
    
    fitness = base_fitness + efficiency_bonus - step_penalty
    
    return fitness

def eval_genomes(genomes, config):
    """Évalue tous les genomes d'une génération."""
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def run_neat():
    """Fonction principale pour entraîner NEAT."""
    # Charger la configuration
    config_path = "config-feedforward.txt"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    # Créer la population
    p = neat.Population(config)
    
    # Ajouter des reporters pour suivre l'évolution
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10))
    
    # Entraîner pendant N générations
    winner = p.run(eval_genomes, 100)
    
    # Sauvegarder le meilleur genome
    with open('winner-frozenlake.pkl', 'wb') as f:
        pickle.dump(winner, f)
    
    print(f'\nMeilleur genome:\n{winner}')
    print(f'Fitness: {winner.fitness}')
    
    # Visualiser le meilleur genome
    plot_genome(winner, config)
    
    return winner, config

def test_winner(genome_path='winner-frozenlake.pkl', config_path='config-feedforward.txt', render=True):
    """Teste le genome gagnant."""
    # Charger le genome et la config
    with open(genome_path, 'rb') as f:
        winner = pickle.load(f)
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    # Créer le réseau
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    # Tester avec rendu
    env = gym.make('FrozenLake-v1', is_slippery=True, render_mode='human' if render else None)
    
    num_tests = 10
    successes = 0
    
    print(f"\nTest du genome gagnant sur {num_tests} épisodes:")
    for i in range(num_tests):
        reward, steps = run_episode(net, env, render=render)
        if reward > 0:
            successes += 1
        print(f"Épisode {i+1}: Récompense={reward}, Steps={steps}")
    
    print(f"\nTaux de succès: {successes}/{num_tests} ({100*successes/num_tests:.1f}%)")
    env.close()
    
    # Visualiser le genome
    plot_genome(winner, config)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Mode test
        test_winner()
    else:
        # Mode entraînement
        winner, config = run_neat()
        
        # Test final
        print("\n" + "="*50)
        print("TEST DU GENOME GAGNANT")
        print("="*50)
        test_winner()