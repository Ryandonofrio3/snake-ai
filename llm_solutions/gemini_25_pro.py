
# solution.py
import numpy as np

# 1. HYPERPARAMETERS
def get_config():
    """
    Returns a dictionary of hyperparameters for the training process.

    - Network Layers: The sizes (24, 16) are chosen to create a 'funnel' effect, 
      forcing the network to learn compressed, abstract representations of the game state. 
      This is large enough to capture complex logic but small enough to be trainable by a GA.
    - Population & Generations: A large population (512) and sufficient generations (300) 
      provide the necessary diversity and time for evolution to occur.
    - GA Parameters: Tournament selection is efficient and provides good selection pressure. 
      Elitism ensures that the best-performing agent is never lost. A low mutation rate 
      with a moderate mutation power allows for fine-tuning without disrupting good solutions.
    """
    return {
        # --- Fixed Network Architecture ---
        "HIDDEN_LAYER_1": 24, # Neurons in the first hidden layer.
        "HIDDEN_LAYER_2": 16, # Neurons in the second hidden layer.

        # --- Training Settings ---
        "POP_SIZE": 512,      # Number of individuals (networks) in the population.
        "GENERATIONS": 300,   # Number of generations to run the training.

        # --- Genetic Algorithm Parameters ---
        "ELITISM_PERCENT": 0.05,     # Percentage of top performers to carry over directly to the next generation.
        "MUTATION_RATE": 0.1,        # Probability that a given weight will be mutated.
        "MUTATION_POWER": 0.4,       # The standard deviation of the Gaussian noise added during mutation.
        "TOURNAMENT_SIZE": 5,        # Number of individuals to select for a tournament.
        "CROSSOVER_ALPHA": 0.5,      # Blend factor for crossover (0.5 for an even average).
    }


# 2. REWARD FUNCTION
def calculate_reward(env, ate_food, is_alive):
    """
    Calculates the fitness score for a snake's performance on one life.

    This function is designed to reward intelligent behavior beyond just surviving.
    It returns a fitness score, not just a step-by-step reward.

    Args:
        env: The game environment object, providing access to `score`, `steps`, etc.
        ate_food (bool): True if the snake ate food on its last move.
        is_alive (bool): False if the snake just died.

    Returns:
        float: The final fitness score for the completed episode.
    """
    # This reward function is called once at the end of an episode (when the snake dies).
    # The goal is to calculate a final "fitness" score for the entire run.
    if is_alive:
        # Should not be called while the snake is still alive, but as a safeguard:
        return 0.0

    # Base reward is the number of steps survived. This encourages staying alive.
    # We use a small multiplier to make it less significant than eating food.
    fitness = env.steps * 0.1

    # Main reward for eating food. This is the primary objective.
    # The reward is exponential to heavily incentivize higher scores.
    # A score of 0 gives 1, 1 gives 10, 2 gives 100, etc. This creates a
    # strong gradient for the GA to climb.
    score = len(env.snake) - 1  # Calculate score from snake length
    fitness += (10 ** score)

    # Penalty for dying: This is implicitly handled. A dead snake's fitness
    # calculation stops, so a snake that dies early will have a naturally low score.
    # No explicit penalty is needed and can sometimes destabilize training.

    # Example Scenarios:
    # 1. Snake lives 50 steps, eats 0 food -> fitness = 50*0.1 + 10^0 = 5 + 1 = 6
    # 2. Snake lives 150 steps, eats 2 food -> fitness = 150*0.1 + 10^2 = 15 + 100 = 115
    # 3. Snake lives 80 steps, eats 2 food -> fitness = 80*0.1 + 10^2 = 8 + 100 = 108
    # This structure correctly values eating food (115 > 108) but also acknowledges
    # that survival is a component of success (108 > 6).

    return fitness


# 3. GENETIC ALGORITHM
def create_ga_instance(num_weights, pop_size):
    """Factory function to create an instance of the GeneticAlgorithm."""
    return GeneticAlgorithm(num_weights, pop_size)

class GeneticAlgorithm:
    """
    Manages the evolution of a population of neural network weights.

    Implements a standard ask-tell interface for evolutionary algorithms.
    - `ask()`: Provides a population of solutions to be evaluated.
    - `tell()`: Updates the internal state based on the fitness of those solutions.

    Features:
        - Elitism: Preserves the best solutions from one generation to the next.
        - Tournament Selection: An efficient method for choosing parents based on fitness.
        - Blended Crossover: Creates offspring by averaging the weights of two parents.
        - Gaussian Mutation: Applies small, random changes to weights.
    """
    def __init__(self, num_weights, pop_size):
        """
        Initializes the Genetic Algorithm.

        Args:
            num_weights (int): The number of weights in a single neural network.
            pop_size (int): The number of individuals in the population.
        """
        self.num_weights = num_weights
        self.pop_size = pop_size
        self.population = np.random.randn(self.pop_size, self.num_weights).astype(np.float32)
        self.fitness = np.zeros(self.pop_size, dtype=np.float32)

    def ask(self):
        """
        Returns the current population of network weights to be evaluated.

        Returns:
            np.ndarray: A (pop_size, num_weights) array of weights.
        """
        return self.population

    def tell(self, fitness_scores):
        """
        Updates the population based on the provided fitness scores from the evaluation.
        This method contains the core logic of the genetic algorithm: selection,
        crossover, and mutation.

        Args:
            fitness_scores (np.ndarray): A 1D array of fitness scores,
                                         corresponding to each individual in the population.
        """
        self.fitness = np.array(fitness_scores, dtype=np.float32)

        # --- Step 1: Elitism ---
        config = get_config()
        elite_count = int(self.pop_size * config["ELITISM_PERCENT"])
        elite_indices = np.argsort(self.fitness)[-elite_count:]
        
        new_population = np.zeros_like(self.population)
        if elite_count > 0:
            new_population[:elite_count] = self.population[elite_indices]

        # --- Step 2: Crossover and Mutation for the rest of the population ---
        for i in range(elite_count, self.pop_size):
            # --- Parent Selection (Tournament) ---
            parent1 = self._tournament_selection(config["TOURNAMENT_SIZE"])
            parent2 = self._tournament_selection(config["TOURNAMENT_SIZE"])

            # --- Crossover (Blended) ---
            child = self._blended_crossover(parent1, parent2, config["CROSSOVER_ALPHA"])
            
            # --- Mutation (Gaussian) ---
            self._mutate(child, config["MUTATION_RATE"], config["MUTATION_POWER"])

            new_population[i] = child

        self.population = new_population

    def _tournament_selection(self, tournament_size):
        """Selects one parent using tournament selection."""
        competitor_indices = np.random.choice(self.pop_size, tournament_size, replace=False)
        competitor_fitness = self.fitness[competitor_indices]
        winner_index = competitor_indices[np.argmax(competitor_fitness)]
        return self.population[winner_index]

    def _blended_crossover(self, parent1, parent2, alpha):
        """Performs blended crossover between two parents."""
        # The child's genes are a weighted average of the parents' genes.
        return alpha * parent1 + (1 - alpha) * parent2
        
    def _mutate(self, individual, mutation_rate, mutation_power):
        """Applies Gaussian mutation to an individual's weights."""
        mutation_mask = np.random.random(size=self.num_weights) < mutation_rate
        noise = np.random.normal(0, mutation_power, size=self.num_weights)
        individual += mutation_mask * noise