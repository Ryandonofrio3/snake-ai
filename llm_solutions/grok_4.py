# solution.py
import numpy as np

# 1. HYPERPARAMETERS
def get_config():
    """
    Return a dictionary of hyperparameters for the fixed 2-layer MLP.
    MUST use these exact key names.
    """
    return {
        "POP_SIZE": 256,
        "GENERATIONS": 500,
        "HIDDEN_LAYER_1": 40,  # Tuned for sufficient capacity to learn Snake patterns
        "HIDDEN_LAYER_2": 20,  # Tuned for compression towards output actions
        # --- Add your GA-specific hyperparameters below ---
        "MUTATION_RATE": 0.005,
        "TOURNAMENT_SIZE": 3,
        "ELITE_RATIO": 0.1,
        "CROSSOVER_RATE": 0.8,
        "MUTATION_SIGMA": 0.5,
    }

# 2. REWARD FUNCTION
def calculate_reward(env, ate_food, is_alive):
    # Per-step reward to encourage eating food quickly without dying.
    # Assumes this is called after each step in the simulation.
    if not is_alive:
        return -100  # Heavy penalty for dying
    if ate_food:
        return 50  # Reward for eating food
    return -1  # Small penalty per step to discourage looping or stalling

# 3. GENETIC ALGORITHM
def create_ga_instance(num_weights, pop_size):
    return GeneticAlgorithm(num_weights, pop_size)

class GeneticAlgorithm:
    def __init__(self, num_weights, pop_size):
        self.num_weights = num_weights
        self.pop_size = pop_size
        # Hardcoded GA hyperparameters (matched to get_config for consistency)
        self.mutation_rate = 0.005
        self.tournament_size = 3
        self.elite_ratio = 0.1
        self.crossover_rate = 0.8
        self.mutation_sigma = 0.5
        
        # Initialize population with random weights ~ N(0, 1)
        self.population = [np.random.randn(num_weights) for _ in range(pop_size)]
        self.fitness = np.zeros(pop_size)

    def ask(self):
        # Return the current population of weight vectors
        return self.population

    def tell(self, rewards):
        # rewards: list of fitness values for the population
        assert len(rewards) == self.pop_size
        
        # Sort population by descending fitness
        sorted_indices = np.argsort(rewards)[::-1]
        sorted_population = [self.population[i] for i in sorted_indices]
        sorted_fitness = [rewards[i] for i in sorted_indices]
        
        # Elitism: carry over top individuals
        elite_size = int(self.elite_ratio * self.pop_size)
        new_population = sorted_population[:elite_size]
        
        # Generate new individuals until population is full
        while len(new_population) < self.pop_size:
            # Select parents via tournament
            parent1 = self._tournament_selection(sorted_population, sorted_fitness)
            parent2 = self._tournament_selection(sorted_population, sorted_fitness)
            
            # Crossover with probability
            if np.random.rand() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1.copy()  # Clone parent1 if no crossover
            
            # Mutate
            child = self._mutate(child)
            
            new_population.append(child)
        
        self.population = new_population

    def _tournament_selection(self, pop, fitness):
        # Select indices for tournament
        candidates = np.random.choice(len(pop), self.tournament_size, replace=False)
        # Pick the one with highest fitness
        best_idx = candidates[np.argmax([fitness[i] for i in candidates])]
        return pop[best_idx]

    def _crossover(self, p1, p2):
        # Uniform crossover
        mask = np.random.rand(self.num_weights) < 0.5
        child = np.where(mask, p1, p2)
        return child

    def _mutate(self, individual):
        # Gaussian mutation
        mutation_mask = np.random.rand(self.num_weights) < self.mutation_rate
        individual[mutation_mask] += np.random.randn(np.sum(mutation_mask)) * self.mutation_sigma
        return individual