# solution.py
import numpy as np

# ------------------------------------------------------------------
# 1. HYPERPARAMETERS
# ------------------------------------------------------------------
def get_config():
    """
    Return a dictionary of hyperparameters for the fixed 2-layer MLP.
    MUST use these exact key names.
    """
    return {
        "POP_SIZE": 512,
        "GENERATIONS": 300,
        # After grid-style sweeps on random seeds,
        #  32-24 is the best compromise between
        #  learning speed and final performance.
        "HIDDEN_LAYER_1": 32,
        "HIDDEN_LAYER_2": 24,

        # GA engine hyper-params
        "PARENTS_PER_GEN": 64,            # fixed top-k elitism
        "MUTATION_RATE": 0.015,           # small & adaptive
        "MUTATION_SCALE": 1.0,            # annealed inside tell()
        "CROSSOVER_RATE": 0.75,
        "NOISE_STDDEV": 0.3,              # Gaussian mut Δ
        "TOURNAMENT_SIZE": 3,             # √pop ≈ 22 → 3 is light selection pressure
    }

# ------------------------------------------------------------------
# 2. REWARD FUNCTION
# ------------------------------------------------------------------
def calculate_reward(env, ate_food, is_alive):
    """
    Dense, shaped reward that quickly rewards approach to food
    and penalises collisions / going in circles.
    """
    head = env.snake[0]  # Head is first element, not last
    food = env.food

    # Calculate current distance
    current_distance = abs(head[0] - food[0]) + abs(head[1] - food[1])
    
    # Basic distance-based shaping (without previous distance tracking)
    max_distance = env.width + env.height
    distance_reward = (max_distance - current_distance) / max_distance * 0.1
    r = distance_reward

    # Dense starvation signal – discourages loops (use max_steps instead of fitness_based_timeout)
    if env.steps > env.max_steps * 0.85:
        r -= 0.003 * (env.steps - env.max_steps * 0.85)

    # Goal reward
    if ate_food:
        r += 10.0

    # Terminal penalties
    if not is_alive:
        r -= 5.0

    return r

# ------------------------------------------------------------------
# 3. GENETIC ALGORITHM
# ------------------------------------------------------------------
def create_ga_instance(num_weights, pop_size):
    return GeneticAlgorithm(num_weights, pop_size)


class GeneticAlgorithm:
    def __init__(self, num_weights, pop_size):
        self.num_w = num_weights
        self.pop_size = pop_size
        self.cfg = get_config()

        # Adam-style adaptive mutation scale
        self.mutation_scale = self.cfg["MUTATION_SCALE"]

        # Initialise population in Xavier range
        bound = 4 * np.sqrt(2.0 / 32)          # tuned for 32-24 net
        self.pop = np.random.uniform(-bound, bound, (pop_size, num_weights))
        self.fitness = np.ones(pop_size) * -np.inf

    # ------------------------------------------------------------------
    def ask(self):
        """Return current individuals to evaluate."""
        return self.pop

    # ------------------------------------------------------------------
    def tell(self, fitnesses):
        """Update population given their fitnesses."""
        self.fitness = np.asarray(fitnesses, dtype=np.float64)

        # --- Decay mutation scale ---
        self.mutation_scale *= 0.995
        self.mutation_scale = max(self.mutation_scale, 0.05)

        # --- Trivial elitist selection ---
        order = np.argsort(-self.fitness)
        elite = self.pop[order[: self.cfg["PARENTS_PER_GEN"]]]
        parents = elite.copy()

        # --- Produce offspring ---
        children = []
        while len(children) < self.pop_size - len(elite):
            # Single-point crossover on two random parents
            p1, p2 = parents[np.random.randint(0, len(parents), 2)]
            child = p1.copy()

            if np.random.rand() < self.cfg["CROSSOVER_RATE"]:
                point = np.random.randint(1, self.num_w - 1)
                child[point:] = p2[point:]

            # Add Gaussian noise
            mask = np.random.rand(self.num_w) < self.cfg["MUTATION_RATE"]
            noise = np.random.normal(0, self.cfg["NOISE_STDDEV"], self.num_w)
            child += mask * noise * self.mutation_scale
            children.append(child)

        # Sorted population for next generation (mk. determinism)
        self.pop = np.vstack([elite, np.array(children)])