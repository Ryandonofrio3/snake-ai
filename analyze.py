# analyze.py
import os
import re
import numpy as np
import matplotlib.pyplot as plt

CHECKPOINT_DIR = "checkpoints"
LEARNING_SPEED_THRESHOLD = 15  # Score to measure how fast an agent learns

def parse_results():
    """Parses checkpoint filenames to extract performance data."""
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

def calculate_metrics(results):
    """Calculates peak performance, learning speed, and robustness."""
    metrics = {}
    for llm, data in results.items():
        # Peak Performance: The absolute best score achieved.
        peak_perf = np.max(data["scores"])

        # Learning Speed: Generations to consistently pass the threshold.
        try:
            # Find the first generation where the score is >= threshold
            speed_idx = np.where(data["scores"] >= LEARNING_SPEED_THRESHOLD)[0][0]
            learning_speed = data["gens"][speed_idx]
        except IndexError:
            learning_speed = "N/A" # Did not reach the threshold

        # Robustness: The score achieved in the final saved generation.
        robustness = data["scores"][-1] if len(data["scores"]) > 0 else 0

        metrics[llm] = {
            "Peak Performance": f"{peak_perf:.2f}",
            "Learning Speed (Gens to 15)": learning_speed,
            "Robustness (Final Score)": f"{robustness:.2f}",
        }
    return metrics

def plot_results(results):
    """Plots the learning curves for all LLMs."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    for llm, data in results.items():
        ax.plot(data["gens"], data["scores"], marker='o', linestyle='-', markersize=4, label=llm)

    ax.set_title("Snake AI: LLM Performance Comparison", fontsize=16)
    ax.set_xlabel("Generations", fontsize=12)
    ax.set_ylabel("Best Food Score", fontsize=12)
    ax.legend(fontsize=10)
    ax.axhline(y=LEARNING_SPEED_THRESHOLD, color='r', linestyle='--', label=f'Speed Threshold ({LEARNING_SPEED_THRESHOLD})')
    plt.tight_layout()
    
    plot_filename = "llm_comparison_plot.png"
    plt.savefig(plot_filename)
    print(f"\n📈 Comparison plot saved to '{plot_filename}'")
    plt.show()

def print_summary_table(metrics):
    """Prints a formatted summary table of the metrics."""
    print("\n--- LLM Performance Summary ---")
    headers = ["LLM Solution", "Peak Perfobomance", "Learning Speed (Gens to 15)", "Robustness (Final Score)"]
    
    # Simple table formatting
    header_str = " | ".join(headers)
    print(header_str)
    print("-" * len(header_str))
    
    for llm, data in metrics.items():
        row = [llm, data["Peak Performance"], str(data["Learning Speed (Gens to 15)"]), data["Robustness (Final Score)"]]
        print(f"{row[0]:<15} | {row[1]:<16} | {row[2]:<27} | {row[3]:<23}")
    print("-" * len(header_str))


if __name__ == "__main__":
    results = parse_results()
    if results:
        metrics = calculate_metrics(results)
        print_summary_table(metrics)
        plot_results(results)