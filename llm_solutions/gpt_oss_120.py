# solution.py
import numpy as np


# ----------------------------------------------------------------------
# 1. HYPER‑PARAMETERS
# ----------------------------------------------------------------------
def get_config():
    """
    Return a dictionary of hyper‑parameters for the fixed 2‑layer MLP.
    The keys must match the names used by the test harness.
    """
    return {
        # ---- GA population settings ----
        "POP_SIZE": 512,          # number of genomes per generation
        "GENERATIONS": 300,       # total generations the harness will run

        # ---- MLP architecture (you may tune these) ----
        "HIDDEN_LAYER_1": 32,     # neurons in first hidden layer
        "HIDDEN_LAYER_2": 20,     # neurons in second hidden layer

        # ---- GA specific hyper‑parameters ----
        "MUTATION_RATE": 0.12,        # probability each weight mutates
        "MUTATION_STD": 0.05,         # std‑dev of Gaussian mutation noise
        "TOURNAMENT_SIZE": 5,         # participants in each tournament
        "ELITISM": 2,                 # number of top individuals copied unchanged
        # you may add more keys if you wish – the harness will ignore the rest
    }


# ----------------------------------------------------------------------
# 2. REWARD FUNCTION
# ----------------------------------------------------------------------
def calculate_reward(env, ate_food, is_alive):
    """
    Compute a scalar reward for the current time‑step.

    Parameters
    ----------
    env : the Snake environment instance (provided by the harness)
    ate_food : bool – True if the agent just ate a piece of food
    is_alive : bool – False if the agent died on this step

    Returns
    -------
    float – the reward for this step
    """
    # small constant reward for staying alive; encourages longer games
    reward = 0.1

    # big positive reward for eating food – proportional to board size
    if ate_food:
        # a simple scale that works for any board size
        reward += (env.width + env.height) * 0.5

    # heavy penalty on death – this pushes the GA to find safe policies
    if not is_alive:
        reward -= 50.0

    return reward


# ----------------------------------------------------------------------
# 3. GENETIC ALGORITHM
# ----------------------------------------------------------------------
def create_ga_instance(num_weights, pop_size):
    """
    Helper required by the harness – builds a GA instance with the
    hyper‑parameters defined in ``get_config``.
    """
    cfg = get_config()
    return GeneticAlgorithm(
        num_weights=num_weights,
        pop_size=pop_size,
        mutation_rate=cfg["MUTATION_RATE"],
        mutation_std=cfg["MUTATION_STD"],
        tournament_size=cfg["TOURNAMENT_SIZE"],
        elitism=cfg["ELITISM"],
    )


class GeneticAlgorithm:
    """
    A simple steady‑state GA with tournament selection, uniform crossover
    and Gaussian mutation.
    """

    def __init__(
        self,
        num_weights,
        pop_size,
        mutation_rate=0.1,
        mutation_std=0.02,
        tournament_size=3,
        elitism=1,
    ):
        self.num_weights = num_weights
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std
        self.tournament_size = tournament_size
        self.elitism = elitism

        # Initialise the population – small random values centred at 0
        self.population = np.random.randn(pop_size, num_weights) * 0.5

    # ------------------------------------------------------------------
    # Public API used by the harness
    # ------------------------------------------------------------------
    def ask(self):
        """
        Return the current population to be evaluated.
        The harness will feed each weight vector into the fixed MLP.
        """
        return self.population.copy()

    def tell(self, fitness):
        """
        Receive a list/array of fitness values (one per individual) and
        evolve the population for the next generation.
        """
        fitness = np.asarray(fitness)
        # Indices of individuals sorted by descending fitness
        sorted_idx = np.argsort(fitness)[::-1]

        # ----------------------------------------------------------------
        # 1️⃣  Elitism – keep the best individuals unchanged
        # ----------------------------------------------------------------
        new_pop = np.empty_like(self.population)
        elite_cnt = min(self.elitism, self.pop_size)
        new_pop[:elite_cnt] = self.population[sorted_idx[:elite_cnt]]

        # ----------------------------------------------------------------
        # 2️⃣  Fill the rest of the new population
        # ----------------------------------------------------------------
        for i in range(elite_cnt, self.pop_size):
            # --- tournament selection for two parents
            p1_idx = self._tournament_select(fitness)
            p2_idx = self._tournament_select(fitness)

            # --- uniform crossover
            child = self._crossover(self.population[p1_idx], self.population[p2_idx])

            # --- mutation
            child = self._mutate(child)

            new_pop[i] = child

        self.population = new_pop

    # ------------------------------------------------------------------
    # Helper methods (private)
    # ------------------------------------------------------------------
    def _tournament_select(self, fitness):
        """
        Choose one parent via tournament selection.
        """
        participants = np.random.choice(self.pop_size, self.tournament_size, replace=False)
        best = participants[np.argmax(fitness[participants])]
        return best

    def _crossover(self, parent1, parent2):
        """
        Uniform crossover – each gene is taken from either parent with
        probability 0.5.
        """
        mask = np.random.rand(self.num_weights) < 0.5
        child = np.where(mask, parent1, parent2)
        return child

    def _mutate(self, individual):
        """
        Apply Gaussian mutation to a subset of genes.
        """
        mutation_mask = np.random.rand(self.num_weights) < self.mutation_rate
        noise = np.random.randn(self.num_weights) * self.mutation_std
        individual = individual + mutation_mask * noise
        return individual