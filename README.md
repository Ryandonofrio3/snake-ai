# LLM Snake AI Benchmark

A comprehensive benchmark testing 11 different LLMs' ability to design genetic algorithms for training neural networks to play Snake. This project evaluates LLMs on algorithmic thinking, hyperparameter tuning, and code generation quality.

##  Results

| Model | Peak Score | Learning Speed | Final Score | Lines of Code |
|-------|------------|----------------|-------------|---------------|
| **GPT-5** | **25.0** | Gen 50 | 24.0 | 368 |
| **DeepSeek-31** | 23.4 | Gen 160 | 23.4 | 284 |
| **Grok Code Fast** | **22.4** | **Gen 80** | 22.4 | **96** |
| **Gemini-25-Pro** | 22.0 | Gen 110 | 17.2 | 178 |
| **Kimi-K2** | 21.6 | Gen 60 | 21.6 | 256 |
| **GPT-OSS-120** | 21.2 | Gen 150 | 21.0 | 178 |
| **Grok-4** | 20.8 | Gen 150 | 19.8 | 105 |
| **Claude Sonnet** | 19.4 | Gen 230 | 16.6 | 178 |
| **Qwen-30B** | 16.0 | Gen 230 | 15.6 | 198 |
| **Claude Opus** | 13.6 | Never | 11.2 | 284 |
| **Mistral Medium** | 0.6 | Never | 0.6 | 254 |

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- [uv](https://github.com/astral-sh/uv) (modern Python package manager)

### Installation
```bash
git clone <your-repo-url>
cd llm-snake-benchmark
uv sync
```

### Run a Single Model
```bash
# Train GPT-5's solution for 500 generations
uv run eval_harness.py --solution gpt_5

# Train with logging
uv run eval_harness.py --solution gpt_5 --log
```

### Run All Models
```bash
# Train all 11 LLM solutions overnight
uv run eval_harness.py --solution all --log
```

### Analyze Results
```bash
# Generate comparison plots and metrics
uv run analyze.py
```

### Watch Tournament
```bash
# Run real-time tournament viewer
uv run multiview.py

# Customize tournament settings
uv run multiview.py --speed 15 --seeds 10 --timeout 300
```

## 📁 Project Structure

```
├── eval_harness.py          # Main evaluation framework
├── xgboost_eval_harness.py  # XGBoost-based evaluation system
├── analyze.py               # Performance analysis & plotting
├── multiview.py             # Real-time tournament viewer
├── llm_solutions/           # All 11 LLM-generated solutions
│   ├── gpt_5.py
│   ├── deepseek_31.py
│   ├── grok_code_fast.py
│   └── ...
├── xgboost_solutions/       # XGBoost-based solutions
│   ├── v3.py                # Advanced curriculum learning example
│   ├── v4.py
│   └── ...
├── checkpoints/             # Saved model weights (generated)
├── xgboost_checkpoints/     # XGBoost model checkpoints (auto-ignored)
└── README.md
```

## 🤖 Adding a New XGBoost Model

The XGBoost evaluation system (`xgboost_eval_harness.py`) allows you to create custom training algorithms for Snake AI using gradient boosting. This system gives you full control over the training process while providing standardized evaluation.

### Prerequisites
- Python 3.8+
- XGBoost library (`uv sync` installs all dependencies)
- Understanding of reinforcement learning concepts

### Required File Structure
Create a new Python file in the `xgboost_solutions/` directory (e.g., `my_model.py`).

### Required Functions

Your solution must implement **exactly three functions**:

#### 1. `get_config()` → Dictionary
Returns hyperparameters and configuration for your training algorithm:
```python
def get_config():
    return {
        "ITERATIONS": 100,                    # Number of training iterations
        "EPISODES_PER_ITERATION": 500,        # Episodes per iteration
        "ELITE_PERCENTILE": 10,               # Top % of episodes to keep
        "GAMMA": 0.95,                        # Discount factor for rewards

        # XGBoost hyperparameters
        "XGB_PARAMS": {
            'objective': 'multi:softprob',
            'num_class': 3,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'max_depth': 6,
            'random_state': 42
        },

        # Custom reward shaping
        "REWARD_FOOD": 1.0,
        "REWARD_DEATH": -1.0,
        "REWARD_SURVIVAL": 0.01
    }
```

#### 2. `calculate_reward(env, ate_food, is_alive)` → Float
**Note:** This is currently a placeholder. The harness uses your reward function through the training algorithm.

#### 3. `create_training_algorithm(config, play_game_func, calculate_reward_func, num_workers, log_func)` → Class Instance
Returns an instance of your training algorithm class:
```python
def create_training_algorithm(config, play_game_func, calculate_reward_func, num_workers, log_func):
    return MyCustomTrainer(config, log_func)
```

### Training Algorithm Class Requirements

Your trainer class must implement:

#### `__init__(self, config, log_func)`
Initialize your algorithm with configuration and logging function.

#### `train_iteration(self, iteration_num)` → Dictionary
This is called once per training iteration. Must return:
```python
{
    'best_score': float,     # Best score achieved this iteration
    'mean_score': float,     # Average score across all episodes
    'model': xgb_model       # Optional: trained XGBoost model for checkpointing
}
```

### Harness-Provided Tools

The harness gives you several helpful functions:

#### `play_game_func(policy_model, calculate_reward_func)`
Plays one complete game and returns trajectory data:
```python
trajectory = play_game_func(model, reward_func)
# Returns list of steps with: obs, action, score, reward, ate_food, is_alive, next_obs
```

#### `log_func(message)`
Standardized logging that works with both console and file output:
```python
log_func(f"Iteration {iteration_num}: Best score = {best_score}")
```

### Example: Study the v3.py Implementation

The `v3.py` file provides an excellent example of a complete XGBoost solution:

```12:15:xgboost_solutions/v3.py
def get_config():
    """
    Final configuration for the Advanced XGBoost Snake AI.
    Uses a lower elite percentile, which is crucial for the bootstrap phase.
    """
    return {
        # --- Core Algorithm Hyperparameters ---
        "ITERATIONS": 150,
        "EPISODES_PER_ITERATION": 1000,
        "ELITE_PERCENTILE": 8, # A lower percentile is critical for early learning
        "GAMMA": 0.98,
```

Key features of v3.py:
- **Curriculum Learning**: Starts on small boards (6x6) and progressively increases difficulty
- **Reward Shaping**: Dense rewards for progress toward food and penalties for walls
- **Elite Selection**: Keeps only top-performing episodes for training
- **Exploration Decay**: Balances exploration vs exploitation over time
- **Bootstrap Phase**: Uses a "golden episode" to initialize learning

### Step-by-Step: Adding Your Model

1. **Create your solution file**:
   ```bash
   touch xgboost_solutions/my_model.py
   ```

2. **Implement the three required functions** (copy structure from v3.py)

3. **Test your model**:
   ```bash
   uv run xgboost_eval_harness.py --solution my_model --log
   ```

4. **Monitor training progress**:
   - Check the generated log file for iteration-by-iteration progress
   - Models are automatically saved to `xgboost_checkpoints/my_model/` when they achieve new best scores

5. **Tune hyperparameters**:
   - Adjust learning rates, episode counts, elite percentiles
   - Experiment with different reward structures
   - Try curriculum learning approaches

### Advanced Features You Can Implement

- **Curriculum Learning**: Start simple, gradually increase difficulty (like v3.py)
- **Reward Shaping**: Dense rewards for intermediate achievements
- **Policy Noise**: Add exploration noise to action probabilities
- **Experience Replay**: Store and reuse past experiences
- **Multi-stage Training**: Different strategies for different phases

### Running Multiple Models

```bash
# Run a specific model
uv run xgboost_eval_harness.py --solution my_model --log

# Run all models in the solutions directory
uv run xgboost_eval_harness.py --solution all --log

# Run models from a custom directory
uv run xgboost_eval_harness.py --solution my_model --solutions_dir my_custom_solutions
```

### Checkpointing & Results

- Models are automatically saved when they achieve new best scores
- Checkpoints are stored in `xgboost_checkpoints/[model_name]/`
- Use the regular `analyze.py` script to compare performance across models
- View trained models compete using `multiview.py`

The XGBoost system gives you complete freedom to experiment with different training algorithms while maintaining fair evaluation standards. Study `v3.py` as a reference implementation, then innovate!

## 🔧 Core Scripts

### `eval_harness.py` - Evaluation Framework
The standardized testing environment that all LLMs compete in:

**Features:**
- Fixed neural network architecture (17 inputs → 2 hidden layers → 3 outputs)
- Standardized Snake environment (10x10 board)
- Fair population sizes and generation counts
- Multi-threaded evaluation for speed
- Automatic checkpointing every 10 generations

**Usage:**
```bash
# Run a specific solution
uv run eval_harness.py --solution gpt_5

# Run all solutions overnight
uv run eval_harness.py --solution all --log
```

### `xgboost_eval_harness.py` - XGBoost Training Framework
Flexible system for implementing custom reinforcement learning algorithms using XGBoost:

**Features:**
- Complete control over training algorithms and hyperparameters
- Standardized Snake environment integration
- Automatic checkpointing of best models
- Multi-worker support for parallel training
- Flexible reward shaping and curriculum learning
- Built-in logging and progress tracking

**Key Differences from eval_harness.py:**
- Uses XGBoost classifiers instead of neural networks
- Requires implementing custom training algorithms
- More flexible reward structures and training strategies
- Better for experimenting with RL algorithms

**Usage:**
```bash
# Run a specific XGBoost solution
uv run xgboost_eval_harness.py --solution v3 --log

# Run all XGBoost solutions
uv run xgboost_eval_harness.py --solution all --log

# Run from custom solutions directory
uv run xgboost_eval_harness.py --solution my_model --solutions_dir my_solutions
```

### `analyze.py` - Performance Analysis
Analyzes training results and generates comparative visualizations:

**Features:**
- Parses checkpoint data from all models
- Calculates learning speed, peak performance, and robustness
- Generates comparison plots
- Prints formatted summary tables

**Output:**
- `llm_comparison_plot.png` - Learning curves visualization
- Terminal summary with key metrics
- Performance rankings across all models

### `multiview.py` - Tournament Viewer
Real-time visualization of multiple trained models competing:

**Features:**
- Side-by-side gameplay of all trained models
- Real-time performance metrics
- Multiple tournament rounds with different seeds
- Speed controls and pause/resume
- Timeout detection for stuck models
- Live leaderboard

**Controls:**
- `SPACE`: Pause/Resume
- `UP/DOWN`: Adjust speed
- `N`: Next round (when all snakes die)
- `ESC`: Quit
