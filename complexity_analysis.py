# complexity_analysis.py
import os
import importlib
import matplotlib.pyplot as plt
import numpy as np

def analyze_code_complexity():
    """Analyze code complexity metrics for each LLM solution"""
    
    results = {}
    performance = {
        'gpt_5': 25.0,
        'deepseek_31': 23.4, 
        'grok_code_fast': 22.4,
        'gemini_25_pro': 22.0,
        'kimi_k2': 21.6,
        'gpt_oss_120': 21.2,
        'grok_4': 20.8,
        'claude_sonnet': 19.4,
        'qwen_30b': 16.0,
        'claude_opus': 13.6,
        'mistral_medium': 0.6
    }
    
    for filename in os.listdir('llm_solutions'):
        if filename.endswith('.py') and not filename.startswith('__'):
            model_name = filename.replace('.py', '')
            
            # Count lines of code
            with open(f'llm_solutions/{filename}', 'r') as f:
                lines = f.readlines()
                # Remove empty lines and comments
                code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
                total_lines = len(lines)
                code_lines_count = len(code_lines)
            
            # Count hyperparameters
            try:
                module = importlib.import_module(f"llm_solutions.{model_name}")
                config = module.get_config()
                hyperparams = len([k for k in config.keys() if k not in ['POP_SIZE', 'GENERATIONS', 'HIDDEN_LAYER_1', 'HIDDEN_LAYER_2']])
            except:
                hyperparams = 0
            
            # Count reward function complexity (lines in calculate_reward)
            reward_lines = 0
            in_reward_func = False
            indent_level = 0
            
            for line in lines:
                if 'def calculate_reward' in line:
                    in_reward_func = True
                    indent_level = len(line) - len(line.lstrip())
                    continue
                
                if in_reward_func:
                    if line.strip() == '':
                        continue
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent <= indent_level and line.strip():
                        break
                    if line.strip() and not line.strip().startswith('#'):
                        reward_lines += 1
            
            results[model_name] = {
                'total_lines': total_lines,
                'code_lines': code_lines_count,
                'hyperparams': hyperparams,
                'reward_complexity': reward_lines,
                'performance': performance.get(model_name, 0),
                'efficiency': performance.get(model_name, 0) / max(code_lines_count, 1)
            }
    
    return results

def create_complexity_visualizations(results):
    """Create visualizations showing complexity vs performance"""
    
    # Extract data
    models = list(results.keys())
    code_lines = [results[m]['code_lines'] for m in models]
    performance = [results[m]['performance'] for m in models]
    efficiency = [results[m]['efficiency'] for m in models]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Performance vs Code Lines
    ax1 = axes[0]
    colors = ['red' if p < 15 else 'orange' if p < 20 else 'green' for p in performance]
    scatter = ax1.scatter(code_lines, performance, c=colors, s=100, alpha=0.7)
    
    # Highlight special models
    for i, model in enumerate(models):
        if model in ['gpt_5', 'grok_code_fast']:
            ax1.annotate(model.replace('_', ' ').title(), 
                        (code_lines[i], performance[i]), 
                        xytext=(10, 10), textcoords='offset points',
                        fontweight='bold', fontsize=10)
        else:
            ax1.annotate(model.replace('_', ' '), 
                        (code_lines[i], performance[i]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
    
    ax1.set_xlabel('Lines of Code')
    ax1.set_ylabel('Peak Performance')
    ax1.set_title('Performance vs Code Complexity')
    ax1.grid(True, alpha=0.3)
    
    # 2. Efficiency (Performance per Line)
    ax2 = axes[1]
    bars = ax2.bar(range(len(models)), efficiency, color=colors, alpha=0.7)
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Performance per Line of Code')
    ax2.set_title('Code Efficiency by Model')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels([m.replace('_', ' ') for m in models], rotation=45, ha='right')
    
    # Highlight top performers
    max_eff_idx = efficiency.index(max(efficiency))
    bars[max_eff_idx].set_color('gold')
    bars[max_eff_idx].set_edgecolor('black')
    bars[max_eff_idx].set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig('complexity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    results = analyze_code_complexity()
    create_complexity_visualizations(results)
