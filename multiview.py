# multiview.py
import pygame
import numpy as np
import argparse
import importlib
import math
import sys
import os
import glob
import random

# We need the SnakeEnv from the harness to run the simulation
from eval_harness import SnakeEnv

class MultiViewer:
    def __init__(self, speed=10, num_seeds=5, timeout=200):
        # Auto-discover all latest checkpoints
        self.agent_files = self.find_latest_checkpoints()
        self.num_agents = len(self.agent_files)
        if self.num_agents == 0:
            print("Error: No agent checkpoint files found.")
            sys.exit(1)

        print(f"Found {self.num_agents} agents with latest checkpoints")
        for i, f in enumerate(self.agent_files):
            print(f"  {i+1}. {f}")

        self.agents = [self.load_agent(f) for f in self.agent_files]
        
        # Use the fixed board size from the harness
        from eval_harness import BOARD_WIDTH, BOARD_HEIGHT
        self.board_w = BOARD_WIDTH
        self.board_h = BOARD_HEIGHT
        
        # Speed and seed controls
        self.speed = speed
        self.num_seeds = num_seeds
        self.current_seed = 0
        self.seeds = [random.randint(1000, 9999) for _ in range(num_seeds)]
        
        # Game state
        self.envs = None
        self.all_dead = False
        self.wins = {agent['name']: 0 for agent in self.agents}
        
        # Timeout settings
        self.max_steps_without_food = timeout  # Max steps without eating food
        self.steps_since_food = []  # Track steps since last food for each agent
        
        # Calculate grid layout
        self.grid_cols = math.ceil(math.sqrt(self.num_agents))
        self.grid_rows = math.ceil(self.num_agents / self.grid_cols)
        self.scale = 25  # Larger scale for better viewing

        pygame.init()
        self.screen_w = self.grid_cols * self.board_w * self.scale
        self.screen_h = self.grid_rows * self.board_h * self.scale + 60  # Extra space for UI
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption("Multi-Snake AI Tournament")
        self.font = pygame.font.SysFont("monospace", 14)
        self.big_font = pygame.font.SysFont("monospace", 18)
        self.clock = pygame.time.Clock()
        
        self.start_new_round()

    def find_latest_checkpoints(self):
        """Find the latest checkpoint for each LLM solution."""
        checkpoint_dirs = glob.glob("checkpoints/*/")
        latest_files = []
        
        for checkpoint_dir in checkpoint_dirs:
            # Find all .npy files in this directory
            npy_files = glob.glob(os.path.join(checkpoint_dir, "*.npy"))
            if npy_files:
                # Sort by modification time and get the latest
                latest_file = max(npy_files, key=os.path.getmtime)
                latest_files.append(latest_file)
                
        return sorted(latest_files)
    
    def start_new_round(self):
        """Start a new round with the current seed."""
        if self.current_seed >= len(self.seeds):
            print(f"\n🏆 TOURNAMENT COMPLETE! 🏆")
            print("Final Results:")
            sorted_wins = sorted(self.wins.items(), key=lambda x: x[1], reverse=True)
            for i, (name, wins) in enumerate(sorted_wins, 1):
                print(f"  {i}. {name}: {wins} wins")
            sys.exit(0)
            
        seed = self.seeds[self.current_seed]
        print(f"\n🎲 Starting round {self.current_seed + 1}/{len(self.seeds)} (seed: {seed})")
        
        # Reset all environments with the same seed
        np.random.seed(seed)
        self.envs = [SnakeEnv(self.board_w, self.board_h) for _ in self.agents]
        for env in self.envs:
            env.reset()
        
        # Reset timeout tracking
        self.steps_since_food = [0] * len(self.agents)
        self.all_dead = False

    def load_agent(self, file_path):
        """Dynamically loads an agent's config, policy, and weights."""
        print(f"Loading agent from: {file_path}")
        try:
            # Extract LLM name from path (e.g., checkpoints/gpt_5/gen_0300_food_24.20.npy)
            # Handle both relative and absolute paths
            path_parts = file_path.replace('\\', '/').split('/')
            checkpoint_idx = None
            for i, part in enumerate(path_parts):
                if part == 'checkpoints':
                    checkpoint_idx = i
                    break
            
            if checkpoint_idx is None or checkpoint_idx + 1 >= len(path_parts):
                raise ValueError("Could not find 'checkpoints' directory in path")
                
            llm_name = path_parts[checkpoint_idx + 1]
            solution_module = importlib.import_module(f"llm_solutions.{llm_name}")
            
            config = solution_module.get_config()
            # Use the harness MLP instead of solution's MLP for consistency
            from eval_harness import MLP, get_config_value
            h1 = get_config_value(config, "HIDDEN_LAYER_1", "hidden1", "H1_UNITS")
            h2 = get_config_value(config, "HIDDEN_LAYER_2", "hidden2", "H2_UNITS")
            policy = MLP(17, h1, h2)  # 17 is the correct input dimension
            weights = np.load(file_path)

            return {"name": llm_name, "policy": policy, "weights": weights, "config": config}
        except Exception as e:
            print(f"Failed to load agent from {file_path}: {e}")
            sys.exit(1)

    def run(self):
        running = True
        paused = False
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_UP and self.speed < 60:
                        self.speed += 5
                    elif event.key == pygame.K_DOWN and self.speed > 1:
                        self.speed -= 5
                    elif event.key == pygame.K_n and self.all_dead:
                        self.current_seed += 1
                        self.start_new_round()

            if not paused and not self.all_dead:
                # Update all living snakes
                alive_count = 0
                for i, (agent, env) in enumerate(zip(self.agents, self.envs)):
                    if env.is_alive:
                        alive_count += 1
                        
                        # Store previous score to check for food eaten
                        prev_score = len(env.snake) - 1
                        
                        obs = env._get_observation()
                        action = agent['policy'].get_action(obs, agent['weights'])
                        # We use the loaded agent's reward function
                        llm_module = importlib.import_module(f"llm_solutions.{agent['name']}")
                        env.step(action, llm_module.calculate_reward)
                        
                        # Check if food was eaten
                        new_score = len(env.snake) - 1
                        if new_score > prev_score:
                            # Food eaten! Reset timeout
                            self.steps_since_food[i] = 0
                        else:
                            # No food eaten, increment timeout
                            self.steps_since_food[i] += 1
                            
                        # Check for timeout (kill snake if it's been too long without food)
                        if self.steps_since_food[i] >= self.max_steps_without_food:
                            env.is_alive = False
                            print(f"⏰ {agent['name']} timed out (no food for {self.max_steps_without_food} steps)")
                            alive_count -= 1
                
                # Check if all are dead
                if alive_count == 0:
                    self.all_dead = True
                    winner = self.find_winner()
                    if winner:
                        self.wins[winner] += 1
                        print(f"🏆 Round {self.current_seed + 1} winner: {winner}")
                    else:
                        print(f"💀 Round {self.current_seed + 1}: All died!")

            self.draw_all()
            pygame.display.flip()
            self.clock.tick(self.speed)

        pygame.quit()
    
    def find_winner(self):
        """Find the snake with the highest score."""
        max_score = -1
        winner = None
        for agent, env in zip(self.agents, self.envs):
            score = len(env.snake) - 1
            if score > max_score:
                max_score = score
                winner = agent['name']
        return winner if max_score > 0 else None

    def draw_all(self):
        """Draw everything: all envs + UI."""
        self.screen.fill((20, 20, 30))  # Dark blue background
        
        # Draw each environment
        for i, (agent, env) in enumerate(zip(self.agents, self.envs)):
            grid_x = i % self.grid_cols
            grid_y = i // self.grid_cols
            offset_x = grid_x * self.board_w * self.scale
            offset_y = grid_y * self.board_h * self.scale
            self.draw_env(env, offset_x, offset_y, agent['name'])
        
        # Draw UI at the bottom
        ui_y = self.grid_rows * self.board_h * self.scale + 10
        
        # Tournament info
        info_text = f"Tournament: Round {self.current_seed + 1}/{len(self.seeds)} | Speed: {self.speed} FPS"
        info_surface = self.font.render(info_text, True, (255, 255, 255))
        self.screen.blit(info_surface, (10, ui_y))
        
        # Controls
        if self.all_dead:
            controls = "SPACE: Pause | UP/DOWN: Speed | N: Next Round | ESC: Quit"
            color = (100, 255, 100)  # Green when waiting
        else:
            controls = "SPACE: Pause | UP/DOWN: Speed | ESC: Quit"
            color = (255, 255, 255)  # White when running
            
        controls_surface = self.font.render(controls, True, color)
        self.screen.blit(controls_surface, (10, ui_y + 20))
        
        # Leaderboard
        leaderboard_text = "Wins: " + " | ".join([f"{name}: {wins}" for name, wins in sorted(self.wins.items(), key=lambda x: x[1], reverse=True)])
        if len(leaderboard_text) > 80:  # Truncate if too long
            leaderboard_text = leaderboard_text[:77] + "..."
        leaderboard_surface = self.font.render(leaderboard_text, True, (200, 200, 255))
        self.screen.blit(leaderboard_surface, (10, ui_y + 40))

    def draw_env(self, env, ox, oy, name):
        s = self.scale
        
        # Border color indicates status
        border_color = (100, 255, 100) if env.is_alive else (255, 100, 100)  # Green alive, red dead
        pygame.draw.rect(self.screen, border_color, (ox, oy, self.board_w * s, self.board_h * s), 2)

        # Draw food
        fx, fy = env.food
        pygame.draw.rect(self.screen, (255, 100, 100), (ox + fx*s, oy + fy*s, s, s))
        
        # Draw snake
        if env.is_alive:
            head_color = (100, 255, 150)
            body_color = (50, 200, 100)
        else:
            head_color = (150, 150, 150)
            body_color = (100, 100, 100)
            
        for i, (x, y) in enumerate(env.snake):
            color = head_color if i == 0 else body_color
            pygame.draw.rect(self.screen, color, (ox + x*s, oy + y*s, s-1, s-1))
        
        # Draw text labels with better visibility
        score = len(env.snake) - 1
        
        # Find this agent's index to get timeout info
        agent_idx = None
        for i, (agent, agent_env) in enumerate(zip(self.agents, self.envs)):
            if agent_env is env:
                agent_idx = i
                break
        
        # Status and timeout info
        if env.is_alive:
            status = "ALIVE"
            status_color = (100, 255, 100)
            timeout_info = f"{self.steps_since_food[agent_idx]}/{self.max_steps_without_food}"
            timeout_color = (255, 255, 100) if self.steps_since_food[agent_idx] > self.max_steps_without_food * 0.7 else (255, 255, 255)
        else:
            status = "DEAD"
            status_color = (255, 100, 100)
            timeout_info = "---"
            timeout_color = (150, 150, 150)
        
        name_surface = self.font.render(name, True, (255, 255, 255))
        score_surface = self.font.render(f"Score: {score}", True, (255, 255, 255))
        status_surface = self.font.render(status, True, status_color)
        timeout_surface = self.font.render(f"Steps: {timeout_info}", True, timeout_color)
        
        self.screen.blit(name_surface, (ox + 5, oy + 5))
        self.screen.blit(score_surface, (ox + 5, oy + 22))
        self.screen.blit(status_surface, (ox + 5, oy + 39))
        self.screen.blit(timeout_surface, (ox + 5, oy + 56))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Snake AI Tournament Viewer")
    parser.add_argument('--speed', type=int, default=10, help='Game speed (FPS, 1-60)')
    parser.add_argument('--seeds', type=int, default=5, help='Number of random seeds to test')
    parser.add_argument('--timeout', type=int, default=200, help='Max steps without food before timeout')
    args = parser.parse_args()

    print("🐍 Multi-Snake AI Tournament 🐍")
    print("Auto-discovering latest checkpoints...")
    
    viewer = MultiViewer(speed=args.speed, num_seeds=args.seeds, timeout=args.timeout)
    
    print(f"\n🎮 Settings:")
    print(f"  Speed: {args.speed} FPS")
    print(f"  Seeds: {args.seeds} rounds")
    print(f"  Timeout: {args.timeout} steps without food")
    print("\n🎮 Controls:")
    print("  SPACE: Pause/Resume")
    print("  UP/DOWN: Adjust speed")  
    print("  N: Next round (when all dead)")
    print("  ESC: Quit")
    print("\nStarting tournament...")
    
    viewer.run()