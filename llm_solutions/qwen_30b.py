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
        "HIDDEN_LAYER_1": 24,
        "HIDDEN_LAYER_2": 16,
        "MUTATION_RATE": 0.1,
        "TOURNAMENT_SIZE": 5,
        "ELITISM_RATE": 0.05
    }

# 2. REWARD FUNCTION
def calculate_reward(env, ate_food, is_alive):
    """
    Calculate reward for the current step based on game state.
    
    Args:
        env: The Snake environment (unused in this implementation)
        ate_food: Boolean indicating if the snake ate food in this step
        is_alive: Boolean indicating if the snake is still alive after this step
    
    Returns:
        float: The reward for this step.
    """
    if ate_food:
        return 10.0      # High reward for eating food
    elif not is_alive:
        return -100.0    # Strong penalty for death
    else:
        return 0.0       # Neutral for normal movement

# 3. GENETIC ALGORITHM
def create_ga_instance(num_weights, pop_size):
    """Create a new GeneticAlgorithm instance with hyperparameters from config."""
    config = get_config()
    return GeneticAlgorithm(
        num_weights=num_weights,
        pop_size=pop_size,
        mutation_rate=config["MUTATION_RATE"],
        tournament_size=config["TOURNAMENT_SIZE"],
        elitism_rate=config["ELITISM_RATE"]
    )

class GeneticAlgorithm:
    """Genetic Algorithm implementation for training Snake AI."""
    
    def __init__(self, num_weights, pop_size, mutation_rate, tournament_size, elitism_rate):
        self.num_weights = num_weights
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_rate = elitism_rate
        self.generation = 0
        self.population = None
        self.current_rewards = None

    def ask(self):
        """Generate next population for evaluation."""
        if self.generation == 0:
            # Initialize population with random weights in [-1, 1]
            self.population = [
                np.random.uniform(-1, 1, self.num_weights)
                for _ in range(self.pop_size)
            ]
            self.generation = 1
            return self.population
        else:
            # Create next generation from current population and rewards
            self.population = self._generate_next_generation()
            self.generation += 1
            return self.population

    def tell(self, rewards):
        """Store rewards for current population and prepare for next generation."""
        self.current_rewards = rewards

    def _generate_next_generation(self):
        """Create next population using selection, crossover, mutation, and elitism."""
        # Selection: Tournament selection
        parents = self._tournament_selection()
        
        # Elitism: Preserve top individuals
        elite_count = int(self.elitism_rate * self.pop_size)
        if elite_count > 0:
            sorted_indices = np.argsort(self.current_rewards)[::-1]
            elites = [self.population[i] for i in sorted_indices[:elite_count]]
        else:
            elites = []
        
        # Create new population (excluding elites)
        new_pop = []
        for _ in range(self.pop_size - elite_count):
            parent1 = parents[np.random.randint(0, self.pop_size)]
            parent2 = parents[np.random.randint(0, self.pop_size)]
            child = self._crossover(parent1, parent2)
            child = self._mutate(child)
            new_pop.append(child)
        
        # Combine elites and new population
        return elites + new_pop

    def _tournament_selection(self):
        """Select parents using tournament selection."""
        parents = []
        n = self.pop_size
        for _ in range(self.pop_size):
            # Randomly select tournament_size individuals
            indices = np.random.choice(n, self.tournament_size, replace=False)
            # Select individual with highest reward
            winner_idx = indices[np.argmax([self.current_rewards[i] for i in indices])]
            parents.append(self.population[winner_idx])
        return parents

    def _crossover(self, parent1, parent2):
        """Uniform crossover between two parents."""
        child = np.zeros_like(parent1)
        for i in range(len(parent1)):
            if np.random.random() < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
        return child

    def _mutate(self, individual):
        """Apply mutation to an individual."""
        for i in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                # Add small random noise (mean 0, std 0.1)
                individual[i] += np.random.normal(0, 0.1)
        return individual