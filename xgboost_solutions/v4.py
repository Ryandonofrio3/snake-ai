# solution_final.py
# The definitive, championship-caliber XGBoost Snake AI.
# This version combines a robust bootstrap, a penalty-free sandbox for early
# learning, and curriculum-based reward shaping for advanced strategy.

import numpy as np
import xgboost as xgb
import random
import time
from collections import deque

# Import SnakeEnv from the harness
from eval_harness import SnakeEnv

# 1. HYPERPARAMETERS
def get_config():
    """
    The final, optimized configuration based on successful experiments.
    """
    return {
        # --- Core Algorithm Hyperparameters ---
        "ITERATIONS": 150,
        "EPISODES_PER_ITERATION": 1000,
        "ELITE_PERCENTILE": 8, # The proven, effective value for bootstrapping
        "GAMMA": 0.98,

        # --- Curriculum Learning ---
        "CURRICULUM_STAGES": [
            {'board_size': (6, 6), 'threshold': 9}, # Threshold for the "sandbox" stage
            {'board_size': (8, 8), 'threshold': 11.5}, # Threshold for the intermediate stage
            {'board_size': (10, 10), 'threshold': 100},# Final challenge
        ],

        # --- Exploration ---
        "EXPLORATION_DECAY": 0.975,
        "INITIAL_EXPLORATION": 0.6,
        "MIN_EXPLORATION": 0.02,
        "POLICY_NOISE_STD": 0.1,

        # --- Shaped Rewards (for stages > 0) ---
        "REWARD_FOOD": 1.0, "REWARD_DEATH": -1.5, "REWARD_PROGRESS": 0.3,
        "REWARD_STAGNATE": -0.05, "REWARD_SURVIVAL": 0.001,
        "REWARD_WALL_PENALTY": -0.015,

        # --- XGBoost Hyperparameters ---
        "XGB_PARAMS": {
            'objective': 'multi:softprob', 'num_class': 3, 'learning_rate': 0.12,
            'n_estimators': 400, 'max_depth': 7, 'subsample': 0.85,
            'colsample_bytree': 0.85, 'reg_alpha': 0.15, 'reg_lambda': 1.2,
            'min_child_weight': 3, 'random_state': 42, 'tree_method': 'hist',
            'eval_metric': 'mlogloss', 'verbosity': 0
        }
    }

# 2. REWARD FUNCTION
def calculate_reward(env, ate_food, is_alive):
    """Placeholder for the harness."""
    return 0.0

class RewarderFinal:
    """
    The proven reward function: a simple sandbox for Stage 0, and
    complex shaping for all later stages.
    """
    def __init__(self, config):
        self.config = config
        self.prev_dist_to_food = float('inf')

    def reset(self):
        self.prev_dist_to_food = None

    def calculate(self, env, ate_food, is_alive):
        # --- Stage 0: The Penalty-Free Sandbox ---
        if env.width <= 6:
            if ate_food: return 1.0
            if not is_alive: return -1.0
            return 0.0 # No other rewards or penalties

        # --- Stages 1+: Advanced Reward Shaping ---
        if not is_alive:
            return self.config["REWARD_DEATH"]
        if ate_food:
            self.prev_dist_to_food = None
            return self.config["REWARD_FOOD"]

        reward = self.config["REWARD_SURVIVAL"]
        head_pos = env.head
        
        is_on_edge = (head_pos[0] == 0 or head_pos[0] == env.width - 1 or
                      head_pos[1] == 0 or head_pos[1] == env.height - 1)
        if is_on_edge:
            reward += self.config["REWARD_WALL_PENALTY"]

        food = env.food
        current_dist_to_food = np.linalg.norm(np.array(head_pos) - np.array(food))
        if self.prev_dist_to_food is not None:
            if current_dist_to_food < self.prev_dist_to_food:
                progress = self.prev_dist_to_food - current_dist_to_food
                reward += self.config["REWARD_PROGRESS"] * progress
            else:
                reward += self.config["REWARD_STAGNATE"]
        self.prev_dist_to_food = current_dist_to_food
        return reward

# 3. TRAINING ALGORITHM
def create_training_algorithm(config, play_game_func, calculate_reward_func, num_workers, log_func):
    return AdvancedXGBoostTrainer(config, log_func)

class AdvancedXGBoostTrainer:
    """
    The final, robust trainer with Pure Imitation Bootstrap and curriculum learning.
    """
    def __init__(self, config, log_func):
        self.config = config; self.log_func = log_func
        self.current_model = None; self.exploration_rate = config["INITIAL_EXPLORATION"]
        self.best_score_ever = 0; self.curriculum_stage = 0
        self._initialize_model()
        board_size = self.config['CURRICULUM_STAGES'][0]['board_size']
        self.log_func("Advanced XGBoost Trainer initialized.")
        self.log_func(f"Starting curriculum at Stage 0: Board Size {board_size}")
    
    def _initialize_model(self):
        dummy_X = np.random.random((10, 17)); dummy_y = np.random.randint(0, 3, 10)
        self.current_model = xgb.XGBClassifier(**self.config["XGB_PARAMS"])
        self.current_model.fit(dummy_X, dummy_y)

    def _create_golden_episode(self):
        self.log_func("Injecting a 'Golden Episode' to bootstrap learning...")
        env = SnakeEnv(6, 6)
        env.snake = deque([(3, 2)]); env.direction = 2; env.food = (3, 3)
        obs = env._get_observation(); action = 1
        return {'states': np.array([obs]), 'actions': np.array([action]),
                'rewards': np.array([1.0]), 'total_reward': 1.0, 'final_score': 1}
    
    def _play_single_episode(self):
        stage_config = self.config["CURRICULUM_STAGES"][self.curriculum_stage]
        width, height = stage_config['board_size']; env = SnakeEnv(width, height)
        obs = env.reset(); done = False
        states, actions, rewards = [], [], []
        rewarder = RewarderFinal(self.config); rewarder.reset()
        while not done:
            action_probs = self.current_model.predict_proba(obs.reshape(1, -1)).flatten()
            noise = np.random.normal(0, self.config["POLICY_NOISE_STD"], 3)
            noisy_probs = action_probs + self.exploration_rate * noise
            action = np.argmax(noisy_probs)
            states.append(obs); actions.append(action)
            obs, _, done, score = env.step(action, lambda *a, **k: 0)
            reward = rewarder.calculate(env, env.ate_food_in_step, env.is_alive)
            rewards.append(reward)
        if not states: return None
        return {'states': np.array(states), 'actions': np.array(actions),
                'rewards': np.array(rewards), 'total_reward': np.sum(rewards),
                'final_score': score}

    def _collect_episodes(self, num_episodes):
        return [ep for _ in range(num_episodes) if (ep := self._play_single_episode()) is not None]

    def _calculate_returns(self, episodes):
        gamma = self.config["GAMMA"]
        for episode in episodes:
            rewards = episode['rewards']; returns = np.zeros_like(rewards, dtype=np.float32)
            running_return = 0.0
            for t in reversed(range(len(rewards))):
                running_return = rewards[t] + gamma * running_return; returns[t] = running_return
            episode['returns'] = returns
        return episodes
    
    def _select_elite_episodes(self, episodes):
        episodes.sort(key=lambda ep: ep['final_score'], reverse=True)
        elite_count = max(1, int(len(episodes) * self.config["ELITE_PERCENTILE"] / 100))
        return episodes[:elite_count]
    
    def _prepare_training_data(self, elite_episodes):
        if not elite_episodes: return np.array([]), np.array([]), np.array([])
        all_states = np.vstack([ep['states'] for ep in elite_episodes])
        all_actions = np.concatenate([ep['actions'] for ep in elite_episodes])
        all_returns = np.concatenate([ep['returns'] for ep in elite_episodes])
        dummy_states = np.zeros((3, 17)); dummy_actions = np.array([0, 1, 2]); dummy_returns = np.zeros(3)
        all_states = np.vstack([all_states, dummy_states])
        all_actions = np.concatenate([all_actions, dummy_actions])
        all_returns = np.concatenate([all_returns, dummy_returns])
        weights = all_returns - np.min(all_returns) + 1e-6 
        return all_states, all_actions, weights
    
    def _train_model(self, X, y, weights):
        if len(X) == 0: return self.current_model
        model = xgb.XGBClassifier(**self.config["XGB_PARAMS"])
        model.fit(X, y, sample_weight=weights, verbose=False)
        return model

    def train_iteration(self, iteration_num):
        start_time = time.time()
        if iteration_num == 1:
            self.log_func("Performing Pure Imitation Bootstrap for Iteration 1...")
            episodes = []; golden_episode = self._create_golden_episode()
            if golden_episode:
                for _ in range(50): episodes.append(golden_episode)
        else:
            episodes = self._collect_episodes(self.config["EPISODES_PER_ITERATION"])
        if not episodes:
            self.log_func("Warning: No episodes collected."); return {'best_score': 0, 'mean_score': 0, 'model': self.current_model}
        
        episodes_with_returns = self._calculate_returns(episodes)
        elite_episodes = self._select_elite_episodes(episodes_with_returns)
        X, y, weights = self._prepare_training_data(elite_episodes)
        
        if len(X) > 0: self.current_model = self._train_model(X, y, weights)
        self.exploration_rate = max(self.config["MIN_EXPLORATION"], self.exploration_rate * self.config["EXPLORATION_DECAY"])
        
        scores = [ep['final_score'] for ep in episodes if ep.get('final_score') is not None]
        if not scores: scores = [0]
        best_score, mean_score = max(scores), np.mean(scores)
        if best_score > self.best_score_ever and iteration_num > 1:
             self.best_score_ever = best_score; self.log_func(f"🎉 New best snake length on this stage: {best_score}")

        stage_config = self.config["CURRICULUM_STAGES"][self.curriculum_stage]
        board_size_str = f"{stage_config['board_size'][0]}x{stage_config['board_size'][1]}"
        self.log_func(f"Iter {iteration_num:02d} [Stage {self.curriculum_stage}, {board_size_str}] | Best: {best_score} | Mean: {mean_score:.2f} | Elite Samples: {len(X)} | Expl: {self.exploration_rate:.3f} | Time: {time.time() - start_time:.2f}s")
        
        if mean_score >= stage_config['threshold']:
            if self.curriculum_stage < len(self.config["CURRICULUM_STAGES"]) - 1:
                self.curriculum_stage += 1
                new_board_size = self.config["CURRICULUM_STAGES"][self.curriculum_stage]['board_size']
                self.log_func("="*60); self.log_func(f"🚀 PROMOTION! Mean score of {mean_score:.2f} reached threshold.")
                self.log_func(f"Moving to curriculum stage {self.curriculum_stage}: Board Size {new_board_size}"); self.log_func("="*60)
                self.best_score_ever = 0 
        
        return {'best_score': best_score, 'mean_score': mean_score, 'model': self.current_model}