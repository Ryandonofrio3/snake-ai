 
# solution.py
import numpy as np
import random

# 1. HYPERPARAMETERS
def get_config():
    """
    Return a dictionary of hyperparameters for the fixed 2-layer MLP.
    MUST use these exact key names.
    """
    return {
        "POP_SIZE": 512,  # Larger population for diversity in GA
        "GENERATIONS": 300,  # Sufficient generations for convergence
        "HIDDEN_LAYER_1": 64,  # Increased for handling complex 17-input state
        "HIDDEN_LAYER_2": 32,  # Balanced to capture interactions without overfitting
        # --- Add your GA-specific hyperparameters below ---
        "MUTATION_RATE": 0.05,  # Lower mutation for stable evolution
        "MUTATION_STRENGTH": 0.1,  # Strength of mutations
        "TOURNAMENT_SIZE": 5,  # Tournament size for selection
        "ELITISM": 10,  # Keep top individuals to preserve best
        "CROSSOVER_RATE": 0.8,  # Probability of crossover
    }

# 2. REWARD FUNCTION
def calculate_reward(env, ate_food, is_alive):
    # Reward based on survival and food consumption
    reward = 0
    if ate_food:
        reward += 100  # High reward for eating food (primary goal)
    if not is_alive:
        reward -= 100  # Heavy penalty for dying
    # Optional: small negative for time to encourage efficiency, but scaled mildly
    reward -= 1  # Per step penalty to discourage loops
    
    return reward

# 3. GENETIC ALGORITHM
def create_ga_instance(num_weights, pop_size):
    return GeneticAlgorithm(num_weights, pop_size)

class GeneticAlgorithm:
    def __init__(self, num_weights, pop_size, mutation_rate=0.05, mutation_strength=0.1,
                 tournament_size=5, elitism=10, crossover_rate=0.8):
        self.num_weights = num_weights
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.crossover_rate = crossover_rate
        self.population = [np.random.randn(num_weights) for _ in range(pop_size)]
        self.fitnesses = np.zeros(pop_size)
    
    def ask(self):
        return self.population.copy()
    
    def tell(self, fitnesses):
        self.fitnesses = np.array(fitnesses)
        # Sort population by fitness (descending)
        sorted_indices = np.argsort(fitnesses)[::-1]
        new_population = []
        
        # Elitism: keep top individuals
        for i in range(self.elitism):
            new_population.append(self.population[sorted_indices[i]])
        
        # Fill rest via selection, crossover, mutation
        while len(new_population) < self.pop_size:
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            child = self._crossover(parent1, parent2)
            child = self._mutate(child)
            new_population.append(child)
        
        self.population = new_population
    
    def _tournament_selection(self):
        # Tournament selection
        candidates = random.sample(range(len(self.fitnesses)), self.tournament_size)
        best_idx = max(candidates, key=lambda i: self.fitnesses[i])
        return self.population[best_idx]
    
    def _crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, self.num_weights - 1)
            child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        else:
            child = parent1.copy()  # Clone if no crossover
        return child
    
    def _mutate(self, individual):
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] += np.random.normal(0, self.mutation_strength)
        return individual