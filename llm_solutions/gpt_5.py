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
        "GENERATIONS": 300,
        "HIDDEN_LAYER_1": 32,  # tuned for capacity vs. search difficulty
        "HIDDEN_LAYER_2": 24,  # tuned for capacity vs. search difficulty

        # --- GA hyperparameters ---
        "SEED": None,                # or an int for reproducibility
        "WEIGHT_INIT_STD": 0.7,      # initial weight std dev
        "WEIGHT_CLIP": 3.0,          # absolute clip for weights after mutation

        "ELITISM": 8,                # number of elites carried over each generation
        "PARENT_POOL_FRACTION": 0.3, # top fraction eligible as parents
        "TOURNAMENT_SIZE": 5,        # tournament size (if used)
        "CROSSOVER": "uniform",      # 'uniform' or 'blend'

        "MUTATION_RATE": 0.08,       # per-gene mutation probability
        "MUTATION_STD": 0.2,         # base mutation std dev (Gaussian)
        "MUTATION_STD_MIN": 0.02,    # floor for decayed std dev
        "MUTATION_DECAY": 0.995,     # multiplicative decay per generation

        # Anti-stagnation (optional but helpful)
        "STAGNATION_GENS": 20,       # if no improvement for these gens -> partial reinit
        "REINIT_FRACTION": 0.2,      # fraction of non-elite population to reinit on stagnation
    }


# 2. REWARD FUNCTION
def calculate_reward(env, ate_food, is_alive):
    """
    Step-wise reward shaping for Snake.
    Robust to different env attribute names; falls back to simple shaping if needed.

    Components:
    - Progress reward: positive if moving closer to food, negative if moving away.
    - Small step penalty to discourage stalling.
    - Big bonus for eating food.
    - Penalty for dying.
    """
    # Safe getters for head/food positions and board size
    def _get_pos(obj, names):
        for n in names:
            if hasattr(obj, n):
                return getattr(obj, n)
        return None

    def _try_snake_head(e):
        # common patterns: env.snake is list (head first), or env.head, or env.snake_head
        if hasattr(e, "snake") and isinstance(e.snake, (list, tuple)) and len(e.snake) > 0:
            head = e.snake[0]
            if isinstance(head, (list, tuple)) and len(head) >= 2:
                return (float(head[0]), float(head[1]))
        for cand in ("head", "snake_head"):
            h = _get_pos(e, (cand,))
            if isinstance(h, (list, tuple)) and len(h) >= 2:
                return (float(h[0]), float(h[1]))
        return None

    def _try_food(e):
        for cand in ("food", "food_pos", "food_position"):
            f = _get_pos(e, (cand,))
            if isinstance(f, (list, tuple)) and len(f) >= 2:
                return (float(f[0]), float(f[1]))
        return None

    def _try_size(e):
        # returns (width, height) if known, else None
        for pair in (("width", "height"), ("w", "h")):
            if hasattr(e, pair[0]) and hasattr(e, pair[1]):
                return float(getattr(e, pair[0])), float(getattr(e, pair[1]))
        if hasattr(e, "grid_size"):
            gs = getattr(e, "grid_size")
            if isinstance(gs, (list, tuple)) and len(gs) >= 2:
                return float(gs[0]), float(gs[1])
        if hasattr(e, "board_size"):
            bs = getattr(e, "board_size")
            if isinstance(bs, (list, tuple)) and len(bs) >= 2:
                return float(bs[0]), float(bs[1])
        return None

    head = _try_snake_head(env)
    food = _try_food(env)
    size = _try_size(env)

    # Default scales
    eat_bonus = 5.0
    death_penalty = -5.0
    step_penalty = -0.01
    progress_scale = 1.0

    # Fallback reward if we can't compute geometry
    fallback_reward = (eat_bonus if ate_food else 0.0) + (death_penalty if not is_alive else step_penalty)

    if head is None or food is None:
        # Cannot compute distance-based shaping
        return fallback_reward

    # Compute normalized Euclidean distance to food
    dx = food[0] - head[0]
    dy = food[1] - head[1]
    curr_dist = float(np.sqrt(dx * dx + dy * dy))

    # Normalization factor: board diagonal if available, else 1.0
    if size is not None:
        max_dist = float(np.sqrt(size[0] * size[0] + size[1] * size[1]))
        if max_dist <= 0:
            max_dist = 1.0
    else:
        max_dist = 1.0

    # Track previous distance in the env instance
    prev_attr = "_prev_food_dist"
    # Reset tracker if new episode likely started (heuristic)
    if hasattr(env, "_episode_step"):
        # Some envs may maintain a step counter; if not, we just rely on attribute existence
        pass

    if not hasattr(env, prev_attr) or ate_food is True:
        # Initialize tracker on first call or right after eating (food relocated)
        setattr(env, prev_attr, curr_dist)

    prev_dist = getattr(env, prev_attr)

    # Progress reward: positive if moving closer, negative if moving away
    delta = prev_dist - curr_dist
    norm_delta = delta / max(max_dist, 1e-6)
    progress_reward = progress_scale * norm_delta

    # Compose total reward
    reward = 0.0
    reward += progress_reward
    reward += eat_bonus if ate_food else 0.0
    reward += step_penalty
    if not is_alive:
        reward += death_penalty

    # Optional mild penalty for prolonged starvation if available
    for attr in ("steps_since_food", "moves_since_last_food", "steps_without_food"):
        if hasattr(env, attr):
            steps_no_food = float(getattr(env, attr))
            # Tiny penalty per step without food
            reward += -1e-4 * steps_no_food
            break

    # Update tracker for next step unless episode ended
    if is_alive:
        setattr(env, prev_attr, curr_dist)
    else:
        # Clean up at death to avoid leaking across episodes
        if hasattr(env, prev_attr):
            delattr(env, prev_attr)

    return float(reward)


# 3. GENETIC ALGORITHM
def create_ga_instance(num_weights, pop_size):
    return GeneticAlgorithm(num_weights, pop_size)


class GeneticAlgorithm:
    """
    Simple, strong baseline GA with:
    - Elitism
    - Truncated parent pool + tournament selection
    - Uniform or blend crossover
    - Gaussian mutation with annealed std dev
    - Partial reinitialization on stagnation
    """

    def __init__(self, num_weights, pop_size):
        cfg = get_config()
        self.num_weights = int(num_weights)
        self.pop_size = int(pop_size)
        # Use provided pop_size, override config's POP_SIZE for consistency
        cfg["POP_SIZE"] = self.pop_size
        self.cfg = cfg

        # RNG
        self.rng = np.random.default_rng(self.cfg.get("SEED", None))

        # Init population
        init_std = float(self.cfg.get("WEIGHT_INIT_STD", 0.7))
        self.population = self.rng.normal(loc=0.0, scale=init_std, size=(self.pop_size, self.num_weights)).astype(np.float32)
        self.population = np.clip(self.population, -self.cfg.get("WEIGHT_CLIP", 3.0), self.cfg.get("WEIGHT_CLIP", 3.0))

        # State
        self.generation = 0
        self.best_fitness = -np.inf
        self.best_genome = self.population[0].copy()
        self.no_improve_gens = 0

        # For ask/tell cycle guard
        self._asked = False

    def ask(self):
        """
        Return current population (genomes) to be evaluated.
        """
        self._asked = True
        # Return a copy to prevent external mutation
        return self.population.copy()

    def tell(self, fitnesses):
        """
        Accept fitness scores for the last asked population and evolve.
        """
        if not self._asked:
            # If tell is called without ask, we assume population is up to date; continue.
            pass
        self._asked = False

        fitnesses = np.asarray(fitnesses, dtype=np.float32).flatten()
        if fitnesses.shape[0] != self.pop_size:
            raise ValueError(f"Expected {self.pop_size} fitness values, got {fitnesses.shape[0]}")

        # Sanitize fitnesses
        fitnesses = np.nan_to_num(fitnesses, nan=-1e9, posinf=1e9, neginf=-1e9)

        # Track best
        gen_best_idx = int(np.argmax(fitnesses))
        gen_best_fit = float(fitnesses[gen_best_idx])
        gen_best = self.population[gen_best_idx].copy()
        if gen_best_fit > self.best_fitness:
            self.best_fitness = gen_best_fit
            self.best_genome = gen_best.copy()
            self.no_improve_gens = 0
        else:
            self.no_improve_gens += 1

        # Elitism
        elite_n = int(self.cfg.get("ELITISM", 8))
        elite_n = max(0, min(elite_n, self.pop_size))
        indices_sorted = np.argsort(-fitnesses)  # descending
        elites = self.population[indices_sorted[:elite_n]].copy()
        elite_fitness = fitnesses[indices_sorted[:elite_n]]

        # Parent pool (truncation selection)
        parent_pool_frac = float(self.cfg.get("PARENT_POOL_FRACTION", 0.3))
        parent_pool_size = max(2, int(parent_pool_frac * self.pop_size))
        parent_pool_size = min(parent_pool_size, self.pop_size)
        parent_pool_idx = indices_sorted[:parent_pool_size]

        # Prepare next population
        next_pop = []
        if elite_n > 0:
            next_pop.append(elites)

        # Breeding
        need = self.pop_size - elite_n
        if need > 0:
            offsprings = self._breed(
                parent_pool_idx=parent_pool_idx,
                need=need,
            )
            next_pop.append(offsprings)

        self.population = np.vstack(next_pop).astype(np.float32)

        # Anti-stagnation: partial reinit if no improvement
        if self.no_improve_gens >= int(self.cfg.get("STAGNATION_GENS", 20)):
            self._partial_reinit(exclude_elites=elite_n)
            self.no_improve_gens = 0  # reset after intervention

        self.generation += 1

    # --------- Internal helpers ---------
    def _breed(self, parent_pool_idx, need):
        """
        Create 'need' offsprings from the parent pool, using crossover + mutation.
        """
        parents = self.population[parent_pool_idx]
        t_size = int(self.cfg.get("TOURNAMENT_SIZE", 5))
        crossover = str(self.cfg.get("CROSSOVER", "uniform")).lower()

        children = np.empty((need, self.num_weights), dtype=np.float32)
        for i in range(need):
            p1 = self._tournament_select(parents, t_size)
            p2 = self._tournament_select(parents, t_size)

            if crossover == "blend":
                child = self._blend_crossover(p1, p2)
            else:
                child = self._uniform_crossover(p1, p2)

            child = self._mutate(child)
            children[i] = child
        return children

    def _tournament_select(self, pool, t_size):
        """
        Tournament selection within 'pool'.
        Pool is a view of current population; we approximate tournament by random sampling.
        Since we don't have pool fitnesses here, we emulate by sampling indices with bias:
        Use a small "softmax rank" approach via index order assumption; alternatively, just random.
        For correctness, we will sample uniformly then pick the one closest to 'best' in pool order.
        """
        # To better reflect fitness, we rely on the fact that 'parents' are already the top subset.
        # So random pick within this pool is acceptable; optionally bias toward earlier indices.
        idxs = self.rng.integers(low=0, high=pool.shape[0], size=t_size)
        # Bias: pick the minimum index (since pool is sorted by fitness descending)
        best_idx = int(np.min(idxs))
        return pool[best_idx]

    def _uniform_crossover(self, p1, p2):
        mask = self.rng.random(self.num_weights) < 0.5
        child = np.where(mask, p1, p2)
        return child.astype(np.float32)

    def _blend_crossover(self, p1, p2, alpha=0.5):
        """
        BLX-alpha-like crossover. For each gene, sample within extended interval.
        """
        lo = np.minimum(p1, p2)
        hi = np.maximum(p1, p2)
        span = hi - lo
        lo_ext = lo - alpha * span
        hi_ext = hi + alpha * span
        u = self.rng.random(self.num_weights)
        child = lo_ext + u * (hi_ext - lo_ext)
        return child.astype(np.float32)

    def _mutate(self, genome):
        mr = float(self.cfg.get("MUTATION_RATE", 0.08))
        base_std = float(self.cfg.get("MUTATION_STD", 0.2))
        min_std = float(self.cfg.get("MUTATION_STD_MIN", 0.02))
        decay = float(self.cfg.get("MUTATION_DECAY", 0.995))
        wclip = float(self.cfg.get("WEIGHT_CLIP", 3.0))

        sigma = max(min_std, base_std * (decay ** self.generation))

        mask = self.rng.random(self.num_weights) < mr
        if not np.any(mask):
            # Ensure at least one mutation for diversity
            idx = self.rng.integers(0, self.num_weights)
            mask[idx] = True

        noise = self.rng.normal(loc=0.0, scale=sigma, size=self.num_weights).astype(np.float32)
        genome = genome + noise * mask.astype(np.float32)
        genome = np.clip(genome, -wclip, wclip)
        return genome

    def _partial_reinit(self, exclude_elites):
        """
        Reinitialize a fraction of the non-elite population to escape stagnation.
        """
        frac = float(self.cfg.get("REINIT_FRACTION", 0.2))
        if frac <= 0.0:
            return
        non_elite_count = self.pop_size - exclude_elites
        if non_elite_count <= 0:
            return
        k = int(max(1, frac * non_elite_count))
        idxs = np.arange(exclude_elites, self.pop_size)
        self.rng.shuffle(idxs)
        chosen = idxs[:k]
        init_std = float(self.cfg.get("WEIGHT_INIT_STD", 0.7))
        self.population[chosen] = self.rng.normal(loc=0.0, scale=init_std, size=(k, self.num_weights)).astype(np.float32)
        self.population[chosen] = np.clip(self.population[chosen], -self.cfg.get("WEIGHT_CLIP", 3.0), self.cfg.get("WEIGHT_CLIP", 3.0))