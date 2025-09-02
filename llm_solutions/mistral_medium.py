import numpy as np
from typing import List, Tuple

# 1. HYPERPARAMETERS
def get_config():
    """
    Return a dictionary of hyperparameters for the fixed 2-layer MLP.
    MUST use these exact key names.
    """
    return {
        "POP_SIZE": 512,               # Large enough for diversity, small enough for efficiency
        "GENERATIONS": 300,             # Enough generations for convergence
        "HIDDEN_LAYER_1": 32,           # First hidden layer size (tuned for feature extraction)
        "HIDDEN_LAYER_2": 16,           # Second hidden layer size (tuned for decision making)
        # --- GA-specific hyperparameters ---
        "MUTATION_RATE": 0.08,          # Balanced between exploration and exploitation
        "MUTATION_SCALE": 0.2,          # Scale of Gaussian mutation
        "TOURNAMENT_SIZE": 7,           # Selection pressure (7 works well for 512 pop)
        "ELITISM_RATIO": 0.05,          # Preserve top 5% of performers
        "CROSSOVER_RATE": 0.7,          # Probability of crossover between parents
        "WEIGHT_CLIP_VALUE": 5.0,       # Prevent extreme weights
        "FITNESS_SMOOTHING": 0.9,       # For exponential moving average of fitness
        "NOVELTY_BONUS": 0.1,           # Bonus for exploring new states
        "DISTANCE_THRESHOLD": 0.2,      # For novelty search
        "MIN_STEPS_BONUS": 50,          # Minimum steps before getting survival bonus
        "FOOD_REWARD": 1.0,             # Base reward for eating food
        "SURVIVAL_REWARD": 0.01,        # Small reward for staying alive
        "DEATH_PENALTY": -0.5,          # Penalty for dying
        "DISTANCE_WEIGHT": 0.3,         # Weight for distance-to-food in reward
        "HEAD_TO_TAIL_WEIGHT": 0.5,     # Weight for head-to-tail distance in reward
    }

# 2. REWARD FUNCTION
def calculate_reward(env, ate_food: bool, is_alive: bool) -> float:
    """
    Calculate reward based on game state and actions.

    Args:
        env: Environment object with current state information
        ate_food: Boolean indicating if food was eaten
        is_alive: Boolean indicating if snake is still alive

    Returns:
        float: Calculated reward value
    """
    config = get_config()
    reward = 0.0

    # Base rewards/penalties
    if ate_food:
        reward += config["FOOD_REWARD"]

    if not is_alive:
        reward += config["DEATH_PENALTY"]
        # Add bonus for longer survival
        if env.steps > config["MIN_STEPS_BONUS"]:
            reward += config["SURVIVAL_REWARD"] * (env.steps - config["MIN_STEPS_BONUS"])
    else:
        reward += config["SURVIVAL_REWARD"]

    # Distance-based rewards (normalized)
    if hasattr(env, 'snake') and hasattr(env, 'food'):
        head = env.snake[0]
        food = env.food
        head_to_food_dist = abs(head[0] - food[0]) + abs(head[1] - food[1])
        max_possible_dist = env.width + env.height
        dist_reward = 1.0 - (head_to_food_dist / max_possible_dist)
        reward += config["DISTANCE_WEIGHT"] * dist_reward

    # Head-to-tail distance reward (encourage not trapping yourself)
    if hasattr(env, 'head_to_tail_dist') and hasattr(env, 'max_possible_dist'):
        tail_reward = env.head_to_tail_dist / env.max_possible_dist
        reward += config["HEAD_TO_TAIL_WEIGHT"] * tail_reward

    # Novelty bonus (if tracking visited states)
    if hasattr(env, 'state_novelty') and env.state_novelty < config["DISTANCE_THRESHOLD"]:
        reward += config["NOVELTY_BONUS"] * (1.0 - env.state_novelty)

    return reward

# 3. GENETIC ALGORITHM
class GeneticAlgorithm:
    def __init__(self, num_weights: int, pop_size: int):
        """
        Initialize the genetic algorithm.

        Args:
            num_weights: Total number of weights in the neural network
            pop_size: Population size
        """
        self.config = get_config()
        self.num_weights = num_weights
        self.pop_size = pop_size
        self.population = []
        self.fitness_history = []
        self.best_fitness = -float('inf')
        self.best_individual = None
        self.smoothing_factor = self.config["FITNESS_SMOOTHING"]

        # Initialize population
        self.population = [np.random.randn(num_weights) * 0.5 for _ in range(pop_size)]

        # Track state novelty (simplified version)
        self.visited_states = set()
        self.state_novelty_cache = {}

    def ask(self) -> List[np.ndarray]:
        """
        Return the current population of weights for evaluation.

        Returns:
            List of weight vectors
        """
        return self.population.copy()

    def _tournament_selection(self, fitness_scores: np.ndarray) -> Tuple[int, int]:
        """
        Select two parents using tournament selection.

        Args:
            fitness_scores: Array of fitness scores for each individual

        Returns:
            Indices of selected parents
        """
        tournament = np.random.choice(len(fitness_scores), self.config["TOURNAMENT_SIZE"], replace=False)
        parent1 = tournament[np.argmax(fitness_scores[tournament])]

        tournament = np.random.choice(len(fitness_scores), self.config["TOURNAMENT_SIZE"], replace=False)
        parent2 = tournament[np.argmax(fitness_scores[tournament])]

        return parent1, parent2

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        Perform uniform crossover between two parents.

        Args:
            parent1: First parent's weights
            parent2: Second parent's weights

        Returns:
            Child weight vector
        """
        if np.random.rand() > self.config["CROSSOVER_RATE"]:
            return parent1.copy() if np.random.rand() < 0.5 else parent2.copy()

        mask = np.random.rand(len(parent1)) < 0.5
        child = parent1.copy()
        child[mask] = parent2[mask]
        return child

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian mutation to an individual.

        Args:
            individual: Weight vector to mutate

        Returns:
            Mutated weight vector
        """
        mutation_mask = np.random.rand(len(individual)) < self.config["MUTATION_RATE"]
        mutation = np.random.randn(len(individual)) * self.config["MUTATION_SCALE"]
        mutated = individual.copy()
        mutated[mutation_mask] += mutation[mutation_mask]

        # Clip weights to prevent extreme values
        np.clip(mutated, -self.config["WEIGHT_CLIP_VALUE"], self.config["WEIGHT_CLIP_VALUE"], out=mutated)

        return mutated

    def _apply_elitism(self, new_population: List[np.ndarray], fitness_scores: np.ndarray) -> List[np.ndarray]:
        """
        Preserve top performers from previous generation.

        Args:
            new_population: New population being constructed
            fitness_scores: Fitness scores of current population

        Returns:
            Population with elitism applied
        """
        num_elites = int(self.config["ELITISM_RATIO"] * self.pop_size)
        elite_indices = np.argsort(fitness_scores)[-num_elites:]
        elites = [self.population[i] for i in elite_indices]

        # Replace worst individuals in new population with elites
        if len(elites) > 0:
            new_population[-len(elites):] = elites

        return new_population

    def tell(self, fitness_scores: List[float]):
        """
        Update the population based on fitness scores.

        Args:
            fitness_scores: List of fitness scores for each individual
        """
        fitness_scores = np.array(fitness_scores)
        smoothed_fitness = np.zeros_like(fitness_scores)

        # Apply fitness smoothing (exponential moving average)
        if len(self.fitness_history) > 0:
            for i in range(len(fitness_scores)):
                smoothed_fitness[i] = (self.smoothing_factor * self.fitness_history[-1][i] +
                                      (1 - self.smoothing_factor) * fitness_scores[i])
        else:
            smoothed_fitness = fitness_scores

        self.fitness_history.append(smoothed_fitness)

        # Track best individual
        best_idx = np.argmax(smoothed_fitness)
        if smoothed_fitness[best_idx] > self.best_fitness:
            self.best_fitness = smoothed_fitness[best_idx]
            self.best_individual = self.population[best_idx].copy()

        # Create new population
        new_population = []

        # Elitism: carry over top performers
        new_population = self._apply_elitism(new_population, smoothed_fitness)

        # Generate offspring until population is filled
        while len(new_population) < self.pop_size:
            # Selection
            parent1_idx, parent2_idx = self._tournament_selection(smoothed_fitness)
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]

            # Crossover
            child = self._crossover(parent1, parent2)

            # Mutation
            child = self._mutate(child)

            new_population.append(child)

        self.population = new_population[:self.pop_size]

def create_ga_instance(num_weights: int, pop_size: int) -> GeneticAlgorithm:
    """
    Factory function to create a GeneticAlgorithm instance.

    Args:
        num_weights: Total number of weights in the neural network
        pop_size: Population size

    Returns:
        GeneticAlgorithm instance
    """
    return GeneticAlgorithm(num_weights, pop_size)