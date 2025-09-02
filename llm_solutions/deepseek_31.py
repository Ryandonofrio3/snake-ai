import numpy as np

# 1. HYPERPARAMETERS
def get_config():
    """
    Return a dictionary of hyperparameters for the fixed 2-layer MLP.
    MUST use these exact key names.
    """
    return {
        "POP_SIZE": 512,
        "GENERATIONS": 300,
        "HIDDEN_LAYER_1": 24,  # Number of neurons in first hidden layer
        "HIDDEN_LAYER_2": 16,  # Number of neurons in second hidden layer
        "MUTATION_RATE": 0.1,  # Probability of mutation per weight
        "TOURNAMENT_SIZE": 5,  # Number of individuals to compete in tournament selection
        "ELITE_COUNT": 2,  # Number of top individuals to carry over unchanged
    }

# 2. REWARD FUNCTION
def calculate_reward(env, ate_food, is_alive):
    """
    Calculate reward for current state.
    env: environment object (may contain additional info)
    ate_food: boolean, whether the snake just ate a food
    is_alive: boolean, whether the snake is still alive
    """
    reward = 0.0
    if ate_food:
        reward += 10.0  # Large reward for eating
    if not is_alive:
        reward -= 10.0  # Large penalty for dying
    else:
        reward += 0.01  # Small reward for staying alive (encourage survival)
    return reward

# 3. GENETIC ALGORITHM
def create_ga_instance(num_weights, pop_size):
    return GeneticAlgorithm(num_weights, pop_size)

class GeneticAlgorithm:
    def __init__(self, num_weights, pop_size):
        self.pop_size = pop_size
        self.num_weights = num_weights
        # Initialize population: each individual is a vector of length num_weights
        self.population = np.random.randn(pop_size, num_weights)
        # GA hyperparameters (could also be passed via config)
        self.mutation_rate = 0.1  # From config, but hardcoded here for simplicity
        self.tournament_size = 5
        self.elite_count = 2  # Number of elites to carry over

    def ask(self):
        """Return the current population (list of candidates)"""
        return self.population

    def tell(self, fitness_scores):
        """
        Apply selection, crossover and mutation to create new population.
        fitness_scores: list of fitness values for each individual in current pop
        """
        # 1. Select parents using tournament selection
        selected_indices = self._tournament_select(fitness_scores, self.pop_size)
        # 2. Create new population via crossover and mutation
        new_population = []
        # Elitism: keep top few individuals unchanged
        elite_indices = np.argsort(fitness_scores)[-self.elite_count:]
        for idx in elite_indices:
            new_population.append(self.population[idx])
        # For the rest, create from selected parents
        for i in range(self.elite_count, self.pop_size):
            parent1 = self.population[selected_indices[i]]
            parent2 = self.population[selected_indices[np.random.randint(0, self.pop_size)]]
            # Crossover: uniform crossover
            child = np.where(np.random.rand(self.num_weights) < 0.5, parent1, parent2)
            # Mutation: add Gaussian noise
            child += np.random.normal(0, self.mutation_rate, self.num_weights)
            new_population.append(child)
        self.population = np.array(new_population)

    def _tournament_select(self, fitness_scores, num_selected):
        """Select individuals via tournament selection"""
        selected = []
        for _ in range(num_selected):
            contestants = np.random.randint(0, len(fitness_scores), self.tournament_size)
            best_index = contestants[np.argmax([fitness_scores[i] for i in contestants])]
            selected.append(best_index)
        return selected