import time
import sys
import os
import argparse
import importlib
import numpy as np
import xgboost as xgb
import glob
import datetime
import traceback
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import joblib
import multiprocessing as mp

# Import the shared Snake Environment from your original harness file
from eval_harness import SnakeEnv, BOARD_WIDTH, BOARD_HEIGHT

# --- HELPER FUNCTIONS ---

def log_message(message, log_file=None):
    """Log message to both console and file if provided."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    print(formatted_message)
    if log_file:
        with open(log_file, "a") as f:
            f.write(formatted_message + "\n")

def get_all_solutions(solutions_dir):
    """Get all available solution files from a specified directory."""
    if not os.path.isdir(solutions_dir):
        return []
    solution_files = glob.glob(os.path.join(solutions_dir, "*.py"))
    solutions = [os.path.basename(f).replace(".py", "") for f in solution_files if not f.endswith("__init__.py")]
    return sorted(solutions)

def get_config_value(config, *keys):
    """Helper to robustly get config values from multiple possible keys."""
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return None

# --- CORE FRAMEWORK FUNCTIONS ---
# These provide the basic building blocks that LLM solutions can use

def play_single_game(policy_model, calculate_reward_func=None):
    """Plays one game and returns trajectory. LLMs can use this in their algorithms."""
    env = SnakeEnv(BOARD_WIDTH, BOARD_HEIGHT)
    obs = env.reset()
    done = False
    trajectory = []
    
    while not done:
        if policy_model:
            try:
                action_probs = policy_model.predict_proba(obs.reshape(1, -1))
                action = np.argmax(action_probs)
            except:
                action = np.random.randint(0, 3)
        else:
            action = np.random.randint(0, 3)
            
        current_obs = obs.copy()
        
        # Use provided reward function or dummy
        reward_func = calculate_reward_func if calculate_reward_func else (lambda e, af, ia: 0)
        obs, reward, done, score = env.step(action, reward_func)
        
        trajectory.append({
            'obs': current_obs, 
            'action': action, 
            'score': score,
            'reward': reward,
            'ate_food': env.ate_food_in_step, 
            'is_alive': env.is_alive,
            'next_obs': obs.copy() if not done else None
        })
        
    return trajectory

# --- MAIN TRAINING SCRIPT ---
# Now LLMs must implement their own training algorithms!

def run_solution(solution_name, solutions_dir, log_file=None):
    """
    Runs the LLM's custom training algorithm for XGBoost Snake AI.
    LLMs must implement their own training loop!
    """
    try:
        solution_module = importlib.import_module(f"{solutions_dir}.{solution_name}")
        
        # Check for required functions
        required_functions = ['get_config', 'calculate_reward', 'create_training_algorithm']
        for func_name in required_functions:
            if not hasattr(solution_module, func_name):
                log_message(f"FATAL: Missing required function '{func_name}' in {solution_name}.py", log_file)
                return False
                
    except ImportError as e:
        log_message(f"FATAL: Could not load '{solutions_dir}/{solution_name}.py': {e}", log_file)
        return False

    try:
        config = solution_module.get_config()
        
        # Get basic configuration
        iterations = get_config_value(config, "ITERATIONS", "iterations")
        if iterations is None:
            iterations = 50  # Default
            
        num_workers = max(1, os.cpu_count() - 2)
        checkpoint_dir = f"xgboost_checkpoints/{solution_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        log_message(f"--- Starting Custom XGBoost Training for {solution_name.upper()} ---", log_file)
        log_message(f"Iterations: {iterations}, Workers: {num_workers}", log_file)
        log_message("-" * 50, log_file)

        # Create the LLM's custom training algorithm instance
        training_algorithm = solution_module.create_training_algorithm(
            config=config,
            play_game_func=play_single_game,
            calculate_reward_func=solution_module.calculate_reward,
            num_workers=num_workers,
            log_func=lambda msg: log_message(msg, log_file)
        )
        
        # Run the LLM's custom training loop
        best_overall_score = -1
        
        for iteration in range(1, iterations + 1):
            t_start = time.time()
            
            
            # Let the LLM's algorithm do one iteration
            try:
                results = training_algorithm.train_iteration(iteration)
                
                if results and 'best_score' in results:
                    best_score_this_iter = results['best_score']
                    mean_score = results.get('mean_score', 0)
                    
                    if best_score_this_iter > best_overall_score:
                        best_overall_score = best_score_this_iter
                        log_message(f"🎉 New best score: {best_overall_score:.2f}", log_file)
                        
                        # Save model if provided
                        if 'model' in results:
                            filename = f"{checkpoint_dir}/best_model_score_{best_overall_score:.2f}.joblib"
                            joblib.dump(results['model'], filename)
                    
                    dt = time.time() - t_start
                    log_message(f"Iter {iteration:02d} | Best: {best_score_this_iter:.2f} | Mean: {mean_score:.2f} | Time: {dt:.2f}s", log_file)
                else:
                    log_message(f"Warning: LLM algorithm returned invalid results for iteration {iteration}", log_file)
                    
            except Exception as e:
                log_message(f"Error in LLM training algorithm iteration {iteration}: {e}", log_file)
                log_message(traceback.format_exc(), log_file)
                return False
        
        log_message(f"--- Training Finished for {solution_name.upper()} ---", log_file)
        log_message(f"Best overall score achieved: {best_overall_score:.2f}", log_file)
        return True
        
    except Exception as e:
        log_message(f"Error running {solution_name}: {str(e)}\n{traceback.format_exc()}", log_file)
        return False

def main():
    parser = argparse.ArgumentParser(description="XGBoost Snake AI Reward-Weighted Training Harness")
    parser.add_argument("--solution", type=str, required=True, help="Name of the solution file in the solutions directory (e.g., 'gemini_pro') or 'all'")
    parser.add_argument("--solutions_dir", type=str, default="xgboost_solutions", help="Directory containing solution files.")
    parser.add_argument("--log", action="store_true", help="Enable logging to a timestamped file.")
    args = parser.parse_args()

    log_file = None
    if args.log:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"xgb_harness_log_{timestamp}.txt"
    
    os.makedirs(args.solutions_dir, exist_ok=True)

    if args.solution.lower() == 'all':
        solutions = get_all_solutions(args.solutions_dir)
        log_message(f"Found {len(solutions)} solutions to run: {', '.join(solutions)}", log_file)
        for sol in solutions:
            run_solution(sol, args.solutions_dir, log_file)
    else:
        run_solution(args.solution, args.solutions_dir, log_file)

if __name__ == '__main__':
    main()