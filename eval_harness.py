# eval_harness.py
import time
import sys
import os
import argparse
import importlib
from collections import deque
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import glob
import datetime
import traceback


BOARD_WIDTH = 10
BOARD_HEIGHT = 10


class MLP:
    """
    The harness now defines the MLP structure to ensure a fair test.
    All LLM solutions will be benchmarked on this exact architecture.
    """
    def __init__(self, in_dim, h1, h2):
        self.in_dim, self.h1, self.h2, self.out_dim = in_dim, h1, h2, 3
        self.num_weights = (in_dim * h1 + h1) + (h1 * h2 + h2) + (h2 * self.out_dim + self.out_dim)

    def get_action(self, obs, weights):
        """Performs a forward pass to determine the next action."""
        idx = 0
        w1 = weights[idx:idx + self.in_dim * self.h1].reshape(self.in_dim, self.h1); idx += self.in_dim * self.h1
        b1 = weights[idx:idx + self.h1]; idx += self.h1
        w2 = weights[idx:idx + self.h1 * self.h2].reshape(self.h1, self.h2); idx += self.h1 * self.h2
        b2 = weights[idx:idx + self.h2]; idx += self.h2
        w3 = weights[idx:idx + self.h2 * self.out_dim].reshape(self.h2, self.out_dim); idx += self.h2 * self.out_dim
        b3 = weights[idx:idx + self.out_dim]
        
        h1_out = np.tanh(obs @ w1 + b1)
        h2_out = np.tanh(h1_out @ w2 + b2)
        logits = h2_out @ w3 + b3
        return np.argmax(logits)

class SnakeEnv:
    """The fixed game environment."""
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.max_steps = width * height * 2
        self.snake, self.direction, self.food = None, None, None
        self.steps, self.is_alive = 0, True
        self.ate_food_in_step = False

    def reset(self):
        self.snake = deque([(self.width // 2, self.height // 2)])
        self.direction = np.random.randint(0, 4)
        self._place_food()
        self.steps, self.is_alive = 0, True
        return self._get_observation()

    @property
    def head(self): return self.snake[0]

    def _place_food(self):
        while True:
            self.food = (np.random.randint(0, self.width), np.random.randint(0, self.height))
            if self.food not in self.snake: break

    def step(self, action: int, reward_func):
        if not self.is_alive: return self._get_observation(), 0.0, True
        self.steps += 1
        self.ate_food_in_step = False
        
        # 0=left, 1=straight, 2=right
        if action == 0: self.direction = (self.direction - 1 + 4) % 4
        elif action == 2: self.direction = (self.direction + 1) % 4
        
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][self.direction]
        new_head = (self.head[0] + dx, self.head[1] + dy)
        
        if (new_head[0] < 0 or new_head[0] >= self.width or new_head[1] < 0 or new_head[1] >= self.height or new_head in self.snake):
            self.is_alive = False
        else:
            self.snake.appendleft(new_head)
            if new_head == self.food:
                self.ate_food_in_step = True
                self._place_food()
            else: self.snake.pop()
            
        reward = reward_func(self, ate_food=self.ate_food_in_step, is_alive=self.is_alive)
        done = not self.is_alive or self.steps >= self.max_steps
        score = len(self.snake) - 1
        return self._get_observation(), reward, done, score

    def _get_observation(self):
        if self.snake is None: return np.zeros(17)
        # 8 wall distances, 2 food vectors, 4 tail vectors, 3 danger vectors
        dirs = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
        wall_dist = [1.0 / (np.linalg.norm(np.array(self.head) - (self.head[0] + d[0]*self.width, self.head[1] + d[1]*self.height)) if 0 <= self.head[0] + d[0] < self.width and 0 <= self.head[1] + d[1] < self.height else 1) for d in dirs]
        food_dx, food_dy = (self.food[0] - self.head[0]) / self.width, (self.food[1] - self.head[1]) / self.height
        wall_dist = np.roll(wall_dist, -2 * self.direction)
        if self.direction == 1:   food_dx, food_dy = food_dy, -food_dx
        elif self.direction == 2: food_dx, food_dy = -food_dx, -food_dy
        elif self.direction == 3: food_dx, food_dy = -food_dy, food_dx
        tail_dir = [0.0] * 4
        if len(self.snake) > 1:
            tx, ty = self.snake[0][0] - self.snake[1][0], self.snake[0][1] - self.snake[1][1]
            if (tx, ty) == (0, -1): tail_dir[0] = 1.0
            if (tx, ty) == (1, 0):  tail_dir[1] = 1.0
            if (tx, ty) == (0, 1):  tail_dir[2] = 1.0
            if (tx, ty) == (-1, 0): tail_dir[3] = 1.0
        danger_fwd, danger_left, danger_right = 0.0, 0.0, 0.0
        x, y = self.head
        fwd_dir_vec = [(0, -1), (1, 0), (0, 1), (-1, 0)][self.direction]
        left_dir_vec = [(1, 0), (0, 1), (-1, 0), (0, -1)][self.direction]
        right_dir_vec = [(-1, 0), (0, -1), (1, 0), (0, 1)][self.direction]
        fwd_pos = (x + fwd_dir_vec[0], y + fwd_dir_vec[1])
        left_pos = (x + left_dir_vec[0], y + left_dir_vec[1])
        right_pos = (x + right_dir_vec[0], y + right_dir_vec[1])
        if not (0 <= fwd_pos[0] < self.width and 0 <= fwd_pos[1] < self.height) or fwd_pos in self.snake: danger_fwd = 1.0
        if not (0 <= left_pos[0] < self.width and 0 <= left_pos[1] < self.height) or left_pos in self.snake: danger_left = 1.0
        if not (0 <= right_pos[0] < self.width and 0 <= right_pos[1] < self.height) or right_pos in self.snake: danger_right = 1.0
        return np.concatenate([wall_dist, [food_dx, food_dy], tail_dir, [danger_fwd, danger_left, danger_right]]).astype(np.float32)


LLM_SOLUTION = None

def init_worker(solution_name):
    """Initializer for each worker process."""
    global LLM_SOLUTION
    LLM_SOLUTION = importlib.import_module(f"llm_solutions.{solution_name}")

def get_config_value(config, *keys):
    """Helper to robustly get config values from multiple possible keys."""
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return None

def get_all_solutions():
    """Get all available solution files."""
    solutions_dir = os.path.join(os.path.dirname(__file__), "llm_solutions")
    solution_files = glob.glob(os.path.join(solutions_dir, "*.py"))
    solutions = []
    for file_path in solution_files:
        filename = os.path.basename(file_path)
        if filename != "__init__.py" and not filename.startswith("_"):
            solution_name = filename.replace(".py", "")
            solutions.append(solution_name)
    return sorted(solutions)

def log_message(message, log_file=None):
    """Log message to both console and file if provided."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    print(formatted_message)
    if log_file:
        try:
            with open(log_file, "a") as f:
                f.write(formatted_message + "\n")
                f.flush()
        except Exception as e:
            print(f"Warning: Failed to write to log file {log_file}: {e}")

def evaluate_genome(weights):
    """Worker function that evaluates a single genome."""
    global LLM_SOLUTION
    config = LLM_SOLUTION.get_config()
    env = SnakeEnv(BOARD_WIDTH, BOARD_HEIGHT)
    obs_dim = env.reset().shape[0]
    
    h1 = get_config_value(config, "HIDDEN_LAYER_1", "hidden1", "H1_UNITS")
    h2 = get_config_value(config, "HIDDEN_LAYER_2", "hidden2", "H2_UNITS")
    episodes = get_config_value(config, "EPISODES_PER_EVAL", "episodes_per_eval")
    if episodes is None:
        episodes = 5  # Default if not specified

    policy = MLP(obs_dim, h1, h2)
    
    total_reward, total_food = 0, 0
    for _ in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.get_action(obs, weights)
            obs, reward, done, score = env.step(action, LLM_SOLUTION.calculate_reward)
            total_reward += reward
        total_food += score
    return total_reward / episodes, total_food / episodes

def evaluate_genome_wrapper(weights):
    """Wrapper for backwards compatibility."""
    return evaluate_genome(weights)

def run_single_solution(solution_name, log_file=None):
    """Run evolution for a single solution."""
    try:
        solution_module = importlib.import_module(f"llm_solutions.{solution_name}")
    except ImportError:
        error_msg = f"Error: Could not find solution file 'llm_solutions/{solution_name}.py'"
        log_message(error_msg, log_file)
        return False
    
    try:
        config = solution_module.get_config()
        num_workers = max(1, os.cpu_count() // 3)  # Use 1/3 of cores to keep cool
        
        temp_env = SnakeEnv(BOARD_WIDTH, BOARD_HEIGHT)
        obs_dim = temp_env.reset().shape[0]

        h1 = get_config_value(config, "HIDDEN_LAYER_1", "hidden1", "H1_UNITS")
        h2 = get_config_value(config, "HIDDEN_LAYER_2", "hidden2", "H2_UNITS")
        
        # Get original values - use original pop_size but standardize generations
        orig_pop_size = get_config_value(config, "POP_SIZE", "pop_size")
        orig_generations = get_config_value(config, "GENERATIONS", "generations", "num_generations")
        
        # USE ORIGINAL POPULATION SIZE BUT STANDARDIZE GENERATIONS
        pop_size = orig_pop_size if orig_pop_size is not None else 256  # Use original population size
        generations = 500  # Fixed generations for all models for fair time comparison
        
        log_message(f"Original config: pop_size={orig_pop_size}, generations={orig_generations}", log_file)
        log_message(f"Using: pop_size={pop_size} (original), generations={generations} (standardized)", log_file)
            
        episodes = get_config_value(config, "EPISODES_PER_EVAL", "episodes_per_eval")
        if episodes is None:
            episodes = 5  # Default if not specified
            
        policy = MLP(obs_dim, h1, h2)
        ga = solution_module.create_ga_instance(policy.num_weights, pop_size)
        checkpoint_dir = f"checkpoints/{solution_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        log_message(f"--- Starting Evolution for {solution_name.upper()} ---", log_file)
        log_message(f"Board Size: {BOARD_WIDTH}x{BOARD_HEIGHT} (Fixed by Harness)", log_file)
        log_message(f"Using {num_workers}/{os.cpu_count()} CPU cores (balanced). Population: {pop_size}", log_file)
        log_message(f"Observation dimension detected: {obs_dim}", log_file)
        log_message(f"Generations: {generations}, Hidden layers: {h1}, {h2}", log_file)
        log_message("-" * 50, log_file)
        
        start_time = time.time()
        best_overall_food = 0.0
        
        for gen in range(1, generations + 1):
            t_start = time.time()
            population = ga.ask()
            
            # Multi-threaded but limited cores to keep cool
            with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(solution_name,)) as executor:
                results = list(executor.map(evaluate_genome_wrapper, population))
            
            fitness_scores, food_scores = np.array(results).T
            ga.tell(fitness_scores)
            
            dt = time.time() - t_start
            best_idx = np.argmax(fitness_scores)
            best_food_this_gen = food_scores[best_idx]
            
            if best_food_this_gen > best_overall_food:
                best_overall_food = best_food_this_gen
            
            progress_msg = f"Gen {gen:03d}/{generations} | Best Food: {best_food_this_gen:.2f} | Mean Food: {np.mean(food_scores):.2f} | Time: {dt:.2f}s"
            log_message(progress_msg, log_file)
            
            if gen % 10 == 0:
                best_weights = population[best_idx]
                filename = f"{checkpoint_dir}/gen_{gen:04d}_food_{best_food_this_gen:.2f}.npy"
                np.save(filename, best_weights)

        total_time = time.time() - start_time
        log_message(f"--- Evolution Finished for {solution_name.upper()} ---", log_file)
        log_message(f"Total time: {total_time/3600:.2f} hours, Best food score: {best_overall_food:.2f}", log_file)
        log_message("", log_file)  # Empty line for separation
        return True
        
    except Exception as e:
        error_msg = f"Error running {solution_name}: {str(e)}\n{traceback.format_exc()}"
        log_message(error_msg, log_file)
        return False

def run_all_solutions(log_file=None):
    """Run evolution for all available solutions."""
    solutions = get_all_solutions()
    
    if not solutions:
        log_message("No solutions found in llm_solutions directory!", log_file)
        return
    
    log_message(f"=== STARTING OVERNIGHT RUN FOR ALL {len(solutions)} SOLUTIONS ===", log_file)
    log_message(f"Solutions to run: {', '.join(solutions)}", log_file)
    log_message("", log_file)
    
    total_start_time = time.time()
    successful_runs = 0
    failed_runs = 0
    
    for i, solution in enumerate(solutions, 1):
        log_message(f">>> Running solution {i}/{len(solutions)}: {solution}", log_file)
        
        if run_single_solution(solution, log_file):
            successful_runs += 1
            log_message(f"✓ {solution} completed successfully", log_file)
        else:
            failed_runs += 1
            log_message(f"✗ {solution} failed", log_file)
        
        log_message("=" * 80, log_file)
    
    total_time = time.time() - total_start_time
    log_message(f"=== OVERNIGHT RUN COMPLETED ===", log_file)
    log_message(f"Total time: {total_time/3600:.2f} hours", log_file)
    log_message(f"Successful runs: {successful_runs}/{len(solutions)}", log_file)
    log_message(f"Failed runs: {failed_runs}/{len(solutions)}", log_file)
    
    if failed_runs > 0:
        log_message("Some solutions failed. Check the log above for details.", log_file)

def main(args):
    # Setup logging
    log_file = None
    if hasattr(args, 'log') and args.log:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"eval_log_{timestamp}.txt"
        log_message(f"Logging to file: {log_file}", log_file)
    
    if args.solution.lower() == 'all':
        run_all_solutions(log_file)
    else:
        if not run_single_solution(args.solution, log_file):
            sys.exit(1)

if __name__ == '__main__':
    if sys.platform.startswith('win'):
        mp.set_start_method('spawn')
    parser = argparse.ArgumentParser(description="LLM Snake AI Evaluation Harness")
    parser.add_argument("--solution", type=str, required=True, 
                       help="Name of the solution file in llm_solutions/ (e.g., 'claude_sonnet') or 'all' to run all solutions")
    parser.add_argument("--log", action="store_true", 
                       help="Enable logging to file with timestamp")
    main(parser.parse_args())