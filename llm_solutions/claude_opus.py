# solution.py
import numpy as np

# 1. HYPERPARAMETERS
def get_config():
    """
    Return a dictionary of hyperparameters for the fixed 2-layer MLP.
    MUST use these exact key names.
    """
    return {
        "POP_SIZE": 512,
        "GENERATIONS": 500,
        "HIDDEN_LAYER_1": 24,  # Sufficient for spatial pattern recognition
        "HIDDEN_LAYER_2": 18,  # Good compression before action selection
        # --- GA-specific hyperparameters ---
        "MUTATION_RATE": 0.15,
        "MUTATION_STRENGTH": 0.3,
        "TOURNAMENT_SIZE": 7,
        "ELITE_SIZE": 25,  # Top 5% preserved
        "CROSSOVER_RATE": 0.75,
        "ADAPTIVE_MUTATION": True,
        "MUTATION_DECAY": 0.995
    }

# 2. REWARD FUNCTION
def calculate_reward(env, ate_food, is_alive):
    """
    Sophisticated reward function that encourages optimal Snake behavior.
    
    Args:
        env: Snake environment with attributes like score, steps, snake_length
        ate_food: Boolean indicating if food was eaten this step
        is_alive: Boolean indicating if snake is still alive
    
    Returns:
        float: Calculated reward
    """
    reward = 0.0
    
    # Primary rewards
    if ate_food:
        # Exponentially increasing reward for consecutive food
        score = len(env.snake) - 1  # Calculate score from snake length
        reward += 1000 * (1 + 0.1 * score)
    
    if is_alive:
        # Small survival bonus to encourage longevity
        reward += 1
        
        # Efficiency bonus: reward based on score/steps ratio
        if env.steps > 0:
            score = len(env.snake) - 1  # Calculate score from snake length
            efficiency = score / max(env.steps, 1)
            reward += efficiency * 50
            
        # Length bonus with diminishing returns
        snake_length = len(env.snake)
        reward += np.sqrt(snake_length) * 10
    else:
        # Death penalties
        base_penalty = -500
        
        # Harsher penalty for early deaths
        if env.steps < 100:
            base_penalty *= 2
            
        # Softer penalty if decent score achieved
        score = len(env.snake) - 1  # Calculate score from snake length  
        if score > 5:
            base_penalty *= 0.5
            
        reward += base_penalty
    
    # Exploration bonus for covering more area (prevents loops)
    if hasattr(env, 'visited_positions'):
        coverage_bonus = len(env.visited_positions) * 0.5
        reward += coverage_bonus
    
    # Distance-to-food bonus (if available)
    if hasattr(env, 'distance_to_food') and is_alive:
        # Inverse distance reward
        max_distance = env.grid_size * 2  # Manhattan distance diagonal
        distance_reward = (max_distance - env.distance_to_food) / max_distance * 20
        reward += distance_reward
    
    return reward

# 3. GENETIC ALGORITHM
def create_ga_instance(num_weights, pop_size):
    """Factory function to create GA instance."""
    config = get_config()
    return GeneticAlgorithm(
        num_weights, 
        pop_size,
        mutation_rate=config.get("MUTATION_RATE", 0.15),
        mutation_strength=config.get("MUTATION_STRENGTH", 0.3),
        tournament_size=config.get("TOURNAMENT_SIZE", 7),
        elite_size=config.get("ELITE_SIZE", 25),
        crossover_rate=config.get("CROSSOVER_RATE", 0.75),
        adaptive_mutation=config.get("ADAPTIVE_MUTATION", True),
        mutation_decay=config.get("MUTATION_DECAY", 0.995)
    )

class GeneticAlgorithm:
    """
    Advanced genetic algorithm with tournament selection, elitism, 
    adaptive mutation, and multiple crossover strategies.
    """
    
    def __init__(self, num_weights, pop_size, mutation_rate=0.15, 
                 mutation_strength=0.3, tournament_size=7, elite_size=25,
                 crossover_rate=0.75, adaptive_mutation=True, mutation_decay=0.995):
        self.num_weights = num_weights
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.initial_mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.crossover_rate = crossover_rate
        self.adaptive_mutation = adaptive_mutation
        self.mutation_decay = mutation_decay
        
        # Initialize population with Xavier/He initialization
        self.population = self._initialize_population()
        self.fitness_scores = None
        self.generation = 0
        self.best_fitness_history = []
        
    def _initialize_population(self):
        """Initialize population with smart weight initialization."""
        population = []
        for _ in range(self.pop_size):
            # Use Xavier initialization for better initial weights
            weights = np.random.randn(self.num_weights) * np.sqrt(2.0 / self.num_weights)
            population.append(weights)
        return np.array(population)
    
    def ask(self):
        """Return current population for evaluation."""
        return self.population
    
    def tell(self, fitness_scores):
        """
        Update population based on fitness scores.
        
        Args:
            fitness_scores: Array of fitness values for each individual
        """
        self.fitness_scores = np.array(fitness_scores)
        self.generation += 1
        
        # Track best fitness
        best_fitness = np.max(self.fitness_scores)
        self.best_fitness_history.append(best_fitness)
        
        # Adaptive mutation rate
        if self.adaptive_mutation:
            self._adapt_mutation_rate()
        
        # Create new population
        new_population = []
        
        # Elitism: preserve top performers
        elite_indices = np.argsort(self.fitness_scores)[-self.elite_size:]
        for idx in elite_indices:
            new_population.append(self.population[idx].copy())
        
        # Fill rest of population
        while len(new_population) < self.pop_size:
            # Tournament selection for parents
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()
            
            # Crossover
            if np.random.random() < self.crossover_rate:
                offspring1, offspring2 = self._crossover(parent1, parent2)
            else:
                offspring1, offspring2 = parent1.copy(), parent2.copy()
            
            # Mutation
            offspring1 = self._mutate(offspring1)
            offspring2 = self._mutate(offspring2)
            
            new_population.append(offspring1)
            if len(new_population) < self.pop_size:
                new_population.append(offspring2)
        
        self.population = np.array(new_population[:self.pop_size])
    
    def _tournament_select(self):
        """Tournament selection."""
        tournament_indices = np.random.choice(
            self.pop_size, 
            size=self.tournament_size, 
            replace=False
        )
        tournament_fitness = self.fitness_scores[tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return self.population[winner_idx].copy()
    
    def _crossover(self, parent1, parent2):
        """
        Advanced crossover with multiple strategies.
        """
        strategy = np.random.choice(['uniform', 'single_point', 'two_point', 'arithmetic'])
        
        if strategy == 'uniform':
            # Uniform crossover
            mask = np.random.random(self.num_weights) < 0.5
            offspring1 = np.where(mask, parent1, parent2)
            offspring2 = np.where(mask, parent2, parent1)
            
        elif strategy == 'single_point':
            # Single-point crossover
            point = np.random.randint(1, self.num_weights)
            offspring1 = np.concatenate([parent1[:point], parent2[point:]])
            offspring2 = np.concatenate([parent2[:point], parent1[point:]])
            
        elif strategy == 'two_point':
            # Two-point crossover
            point1 = np.random.randint(0, self.num_weights - 1)
            point2 = np.random.randint(point1 + 1, self.num_weights)
            offspring1 = parent1.copy()
            offspring2 = parent2.copy()
            offspring1[point1:point2] = parent2[point1:point2]
            offspring2[point1:point2] = parent1[point1:point2]
            
        else:  # arithmetic
            # Arithmetic crossover
            alpha = np.random.random()
            offspring1 = alpha * parent1 + (1 - alpha) * parent2
            offspring2 = (1 - alpha) * parent1 + alpha * parent2
        
        return offspring1, offspring2
    
    def _mutate(self, individual):
        """
        Advanced mutation with multiple strategies.
        """
        if np.random.random() < self.mutation_rate:
            strategy = np.random.choice(['gaussian', 'uniform', 'reset'], p=[0.7, 0.2, 0.1])
            
            if strategy == 'gaussian':
                # Gaussian mutation
                mutation_mask = np.random.random(self.num_weights) < 0.3
                noise = np.random.randn(self.num_weights) * self.mutation_strength
                individual[mutation_mask] += noise[mutation_mask]
                
            elif strategy == 'uniform':
                # Uniform mutation
                mutation_mask = np.random.random(self.num_weights) < 0.2
                noise = np.random.uniform(-self.mutation_strength, self.mutation_strength, self.num_weights)
                individual[mutation_mask] += noise[mutation_mask]
                
            else:  # reset
                # Reset random weights
                num_reset = max(1, int(0.05 * self.num_weights))
                reset_indices = np.random.choice(self.num_weights, num_reset, replace=False)
                individual[reset_indices] = np.random.randn(num_reset) * np.sqrt(2.0 / self.num_weights)
            
            # Clip weights to prevent explosion
            individual = np.clip(individual, -5, 5)
        
        return individual
    
    def _adapt_mutation_rate(self):
        """Adapt mutation rate based on fitness improvement."""
        if len(self.best_fitness_history) > 10:
            # Check fitness improvement over last 10 generations
            recent_improvement = self.best_fitness_history[-1] - self.best_fitness_history[-10]
            
            if recent_improvement < 0.01:  # Stagnation
                # Increase mutation to escape local optima
                self.mutation_rate = min(0.3, self.mutation_rate * 1.1)
                self.mutation_strength = min(0.5, self.mutation_strength * 1.05)
            else:
                # Decay mutation for fine-tuning
                self.mutation_rate *= self.mutation_decay
                self.mutation_strength *= self.mutation_decay
        
        # Ensure minimum mutation
        self.mutation_rate = max(0.01, self.mutation_rate)
        self.mutation_strength = max(0.05, self.mutation_strength)