# analyze.py
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob

CHECKPOINT_DIR = "checkpoints"
XGBOOST_CHECKPOINT_DIR = "xgboost_checkpoints"
LEARNING_SPEED_THRESHOLD = 15  # Score to measure how fast an agent learns

def parse_llm_results():
    """Parses LLM checkpoint filenames to extract performance data."""
    results = {}
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"Error: Checkpoint directory '{CHECKPOINT_DIR}' not found.")
        return None

    for llm_name in os.listdir(CHECKPOINT_DIR):
        llm_path = os.path.join(CHECKPOINT_DIR, llm_name)
        if not os.path.isdir(llm_path):
            continue

        gens, scores = [], []
        gen_to_best_score = {}  # Track best score per generation
        
        for filename in sorted(os.listdir(llm_path)):
            match = re.match(r"gen_(\d+)_food_([\d\.]+)\.npy", filename)
            if match:
                gen = int(match.group(1))
                score = float(match.group(2))
                
                # Keep only the best score for each generation
                if gen not in gen_to_best_score or score > gen_to_best_score[gen]:
                    gen_to_best_score[gen] = score
        
        # Convert to sorted lists
        for gen in sorted(gen_to_best_score.keys()):
            gens.append(gen)
            scores.append(gen_to_best_score[gen])

        if gens:
            results[llm_name] = {"gens": np.array(gens), "scores": np.array(scores)}
    return results

def parse_xgboost_results(solutions=None):
    """Parses XGBoost checkpoint filenames to extract performance data."""
    results = {}
    if not os.path.exists(XGBOOST_CHECKPOINT_DIR):
        print(f"Error: XGBoost checkpoint directory '{XGBOOST_CHECKPOINT_DIR}' not found.")
        return None

    # If specific solutions are provided, only analyze those
    if solutions:
        solution_dirs = [os.path.join(XGBOOST_CHECKPOINT_DIR, sol) for sol in solutions]
    else:
        solution_dirs = glob.glob(os.path.join(XGBOOST_CHECKPOINT_DIR, "*"))

    for solution_path in solution_dirs:
        if not os.path.isdir(solution_path):
            continue
            
        solution_name = os.path.basename(solution_path)
        scores = []
        
        # Parse XGBoost checkpoint files (e.g., best_model_score_11.00.joblib)
        for filename in sorted(os.listdir(solution_path)):
            match = re.match(r"best_model_score_([\d\.]+)\.joblib", filename)
            if match:
                score = float(match.group(1))
                scores.append(score)

        if scores:
            # For XGBoost, we don't have generations, so we'll use iteration numbers
            iterations = list(range(1, len(scores) + 1))
            results[solution_name] = {"gens": np.array(iterations), "scores": np.array(scores)}
    
    return results

def calculate_metrics(results, xgboost_mode=False):
    """Calculates peak performance, learning speed, and robustness."""
    metrics = {}
    for solution, data in results.items():
        # Peak Performance: The absolute best score achieved.
        peak_perf = np.max(data["scores"])

        # Learning Speed: Generations/iterations to consistently pass the threshold.
        try:
            # Find the first generation/iteration where the score is >= threshold
            speed_idx = np.where(data["scores"] >= LEARNING_SPEED_THRESHOLD)[0][0]
            learning_speed = data["gens"][speed_idx]
        except IndexError:
            learning_speed = "N/A" # Did not reach the threshold

        # Robustness: The score achieved in the final saved generation/iteration.
        robustness = data["scores"][-1] if len(data["scores"]) > 0 else 0

        if xgboost_mode:
            speed_key = "Learning Speed (Iters to 15)"
        else:
            speed_key = "Learning Speed (Gens to 15)"

        metrics[solution] = {
            "Peak Performance": f"{peak_perf:.2f}",
            speed_key: learning_speed,
            "Robustness (Final Score)": f"{robustness:.2f}",
        }
    return metrics

def plot_results(results, xgboost_mode=False):
    """Plots the learning curves for all solutions."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    for solution, data in results.items():
        ax.plot(data["gens"], data["scores"], marker='o', linestyle='-', markersize=4, label=solution)

    if xgboost_mode:
        title = "Snake AI: XGBoost Performance Comparison"
        xlabel = "Training Iterations"
        plot_filename = "xgboost_comparison_plot.png"
    else:
        title = "Snake AI: LLM Performance Comparison"
        xlabel = "Generations"
        plot_filename = "llm_comparison_plot.png"

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Best Food Score", fontsize=12)
    ax.legend(fontsize=10)
    ax.axhline(y=LEARNING_SPEED_THRESHOLD, color='r', linestyle='--', label=f'Speed Threshold ({LEARNING_SPEED_THRESHOLD})')
    plt.tight_layout()
    
    plt.savefig(plot_filename)
    print(f"\n📈 Comparison plot saved to '{plot_filename}'")
    plt.show()

def print_summary_table(metrics, xgboost_mode=False):
    """Prints a formatted summary table of the metrics."""
    if xgboost_mode:
        print("\n--- XGBoost Performance Summary ---")
        headers = ["XGBoost Solution", "Peak Performance", "Learning Speed (Iters to 15)", "Robustness (Final Score)"]
    else:
        print("\n--- LLM Performance Summary ---")
        headers = ["LLM Solution", "Peak Performance", "Learning Speed (Gens to 15)", "Robustness (Final Score)"]
    
    # Simple table formatting
    header_str = " | ".join(headers)
    print(header_str)
    print("-" * len(header_str))
    
    for solution, data in metrics.items():
        speed_key = "Learning Speed (Iters to 15)" if xgboost_mode else "Learning Speed (Gens to 15)"
        row = [solution, data["Peak Performance"], str(data[speed_key]), data["Robustness (Final Score)"]]
        print(f"{row[0]:<15} | {row[1]:<16} | {row[2]:<27} | {row[3]:<23}")
    print("-" * len(header_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snake AI Performance Analysis")
    parser.add_argument('--xg', action='store_true', help='Analyze XGBoost models instead of LLM genetic algorithm models')
    parser.add_argument('--solutions', type=str, nargs='+', 
                        help='List of solution names to analyze (optional for XGBoost mode, analyzes all if not specified)')
    args = parser.parse_args()

    if args.xg:
        print("🐍 XGBoost Snake AI Performance Analysis 🐍")
        results = parse_xgboost_results(args.solutions)
        mode_name = "XGBoost"
    else:
        print("🐍 LLM Genetic Algorithm Snake AI Performance Analysis 🐍")
        results = parse_llm_results()
        mode_name = "LLM Genetic Algorithm"
    
    if results:
        print(f"Analyzing {mode_name} results...")
        if args.solutions and args.xg:
            print(f"Solutions: {', '.join(args.solutions)}")
        
        metrics = calculate_metrics(results, xgboost_mode=args.xg)
        print_summary_table(metrics, xgboost_mode=args.xg)
        plot_results(results, xgboost_mode=args.xg)
    else:
        print(f"No {mode_name} results found to analyze.")