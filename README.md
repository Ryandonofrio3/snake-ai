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
├── eval_harness.py     # Main evaluation framework
├── analyze.py          # Performance analysis & plotting
├── multiview.py        # Real-time tournament viewer
├── llm_solutions/      # All 11 LLM-generated solutions
│   ├── gpt_5.py
│   ├── deepseek_31.py
│   ├── grok_code_fast.py
│   └── ...
├── checkpoints/        # Saved model weights (generated)
└── README.md
```

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
