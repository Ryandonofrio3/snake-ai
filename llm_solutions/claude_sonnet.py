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
        "HIDDEN_LAYER_1": 32,  # Good capacity for 17 inputs
        "HIDDEN_LAYER_2": 16,  # Tapering down toward 3 outputs
        # --- GA-specific hyperparameters ---
        "MUTATION_RATE": 0.15,
        "MUTATION_STRENGTH": 0.3,
        "TOURNAMENT_SIZE": 7,
        "ELITE_SIZE": 10,  # Number of best individuals to keep
        "CROSSOVER_RATE": 0.8,
    }

# 2. REWARD FUNCTION
def calculate_reward(env, ate_food, is_alive):
    """
    Calculate reward for the current state.
    
    Args:
        env: Snake environment with attributes like snake, food, etc.
        ate_food: Boolean indicating if food was eaten this step
        is_alive: Boolean indicating if snake is still alive
    
    Returns:
        float: Reward value
    """
    reward = 0.0
    
    if not is_alive:
        # Heavy penalty for dying
        return -100.0
    
    if ate_food:
        # Large reward for eating food, scaled by current length
        reward += 50.0 + len(env.snake) * 5.0
    else:
        # Small reward for staying alive
        reward += 1.0
        
        # Distance-based reward to encourage moving toward food
        if hasattr(env, 'snake') and hasattr(env, 'food'):
            head = env.snake[0]
            food = env.food
            
            # Manhattan distance to food
            distance = abs(head[0] - food[0]) + abs(head[1] - food[1])
            max_distance = env.width + env.height
            
            # Reward getting closer to food (small but helpful)
            proximity_reward = 5.0 * (1.0 - distance / max_distance)
            reward += proximity_reward
    
    # Bonus for length (encourage growth)
    if hasattr(env, 'snake'):
        length_bonus = len(env.snake) * 0.5
        reward += length_bonus
    
    return reward

# 3. GENETIC ALGORITHM
def create_ga_instance(num_weights, pop_size):
    return GeneticAlgorithm(num_weights, pop_size)

class GeneticAlgorithm:
    def __init__(self, num_weights, pop_size):
        self.num_weights = num_weights
        self.pop_size = pop_size
        self.generation = 0
        
        # Get config for hyperparameters
        config = get_config()
        self.mutation_rate = config["MUTATION_RATE"]
        self.mutation_strength = config["MUTATION_STRENGTH"]
        self.tournament_size = config["TOURNAMENT_SIZE"]
        self.elite_size = config["ELITE_SIZE"]
        self.crossover_rate = config["CROSSOVER_RATE"]
        
        # Initialize population with Xavier/Glorot initialization
        self.population = np.random.normal(0, 0.5, (pop_size, num_weights))
        self.fitness_scores = np.zeros(pop_size)
        self.best_fitness = -float('inf')
        self.best_individual = None
        
    def ask(self):
        """Return the current population for evaluation."""
        return self.population.copy()
    
    def tell(self, fitness_scores):
        """Update the population based on fitness scores."""
        self.fitness_scores = np.array(fitness_scores)
        
        # Track best individual
        best_idx = np.argmax(self.fitness_scores)
        if self.fitness_scores[best_idx] > self.best_fitness:
            self.best_fitness = self.fitness_scores[best_idx]
            self.best_individual = self.population[best_idx].copy()
        
        # Create next generation
        new_population = np.zeros_like(self.population)
        
        # Elitism: Keep best individuals
        elite_indices = np.argsort(self.fitness_scores)[-self.elite_size:]
        new_population[:self.elite_size] = self.population[elite_indices]
        
        # Generate rest of population through selection, crossover, and mutation
        for i in range(self.elite_size, self.pop_size):
            if np.random.random() < self.crossover_rate:
                # Crossover
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                child = self._crossover(parent1, parent2)
            else:
                # Just mutation of selected parent
                child = self._tournament_selection().copy()
            
            # Mutation
            child = self._mutate(child)
            new_population[i] = child
        
        self.population = new_population
        self.generation += 1
    
    def _tournament_selection(self):
        """Select individual using tournament selection."""
        tournament_indices = np.random.choice(
            self.pop_size, self.tournament_size, replace=False
        )
        tournament_fitness = self.fitness_scores[tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return self.population[winner_idx].copy()
    
    def _crossover(self, parent1, parent2):
        """Uniform crossover between two parents."""
        mask = np.random.random(self.num_weights) < 0.5
        child = np.where(mask, parent1, parent2)
        return child
    
    def _mutate(self, individual):
        """Apply Gaussian mutation to individual."""
        mutation_mask = np.random.random(self.num_weights) < self.mutation_rate
        mutations = np.random.normal(0, self.mutation_strength, self.num_weights)
        individual[mutation_mask] += mutations[mutation_mask]
        
        # Clip weights to prevent explosion
        individual = np.clip(individual, -5.0, 5.0)
        
        return individual