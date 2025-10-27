"""
Market Simulation Module for DQN Price Competition

This module orchestrates the complete simulation process including training,
evaluation, and metrics calculation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime
import os
import json

from logit_market_env import LogitBertrandMarket, create_market_from_config
from dqn_agents import DQNAgent
from experiment_config import (
    get_dqn_config_for_experiment,
    interpret_metrics,
    TRAINING_CONFIG,
    EVALUATION_METRICS
)


class MarketSimulation:
    """
    Orchestrates multi-agent DQN simulation in Logit Bertrand market
    """
    
    def __init__(
        self,
        experiment_config: Dict,
        learning_rate: Optional[float] = None,
        save_dir: str = "results",
        verbose: bool = True,
    ):
        """
        Initialize market simulation
        
        Args:
            experiment_config: Experiment configuration dictionary
            learning_rate: Optional specific learning rate
            save_dir: Directory to save results
            verbose: Whether to print progress
        """
        self.experiment_config = experiment_config
        self.experiment_name = experiment_config['name']
        self.verbose = verbose
        
        # Create save directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(save_dir, f"{self.experiment_name}_{timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize market environment
        self.env = create_market_from_config(experiment_config)
        self.n_firms = self.env.n_firms
        
        # Get DQN configuration
        self.dqn_config = get_dqn_config_for_experiment(experiment_config, learning_rate)
        
        # Initialize agents
        self.agents = []
        for i in range(self.n_firms):
            agent = DQNAgent(
                state_size=self.env.observation_space.shape[0],
                action_size=self.env.action_space.n,
                config=self.dqn_config,
                agent_id=i,
                seed=42 + i
            )
            self.agents.append(agent)
        
        # Get benchmark values
        self._setup_benchmarks()
        
        # Initialize tracking variables
        self.training_history = {
            'episode_rewards': [[] for _ in range(self.n_firms)],
            'episode_prices': [[] for _ in range(self.n_firms)],
            'episode_profits': [[] for _ in range(self.n_firms)],
            'episode_shares': [[] for _ in range(self.n_firms)],
            'rpdi': [],
            'delta': [],
        }
        
        self.evaluation_results = None
        
        if self.verbose:
            print(f"Initialized simulation: {self.experiment_name}")
            print(f"Number of firms: {self.n_firms}")
            print(f"Learning rate: {self.dqn_config['learning_rate']}")
            print(f"Substitutability: {experiment_config['market_parameters']['substitutability']}")
    
    def _setup_benchmarks(self):
        """Setup benchmark values for metrics calculation"""
        benchmarks = self.experiment_config['benchmarks']
        
        # Handle both symmetric and asymmetric cases
        if 'nash_prices' in benchmarks:
            # Asymmetric case
            self.nash_prices = np.array(benchmarks['nash_prices'])
            self.monopoly_prices = np.array(benchmarks['monopoly_prices'])
            self.nash_profits = np.array(benchmarks['nash_profits'])
            self.monopoly_profits = np.array(benchmarks['monopoly_profits'])
            
            # Use averages for overall metrics
            self.nash_price_avg = benchmarks['nash_price_avg']
            self.monopoly_price_avg = benchmarks['monopoly_price_avg']
            self.nash_profit_avg = benchmarks['nash_profit_avg']
            self.monopoly_profit_avg = benchmarks['monopoly_profit_avg']
        else:
            # Symmetric case
            self.nash_prices = np.ones(self.n_firms) * benchmarks['nash_price']
            self.monopoly_prices = np.ones(self.n_firms) * benchmarks['monopoly_price']
            self.nash_profits = np.ones(self.n_firms) * benchmarks['nash_profit']
            self.monopoly_profits = np.ones(self.n_firms) * benchmarks['monopoly_profit']
            
            self.nash_price_avg = benchmarks['nash_price']
            self.monopoly_price_avg = benchmarks['monopoly_price']
            self.nash_profit_avg = benchmarks['nash_profit']
            self.monopoly_profit_avg = benchmarks['monopoly_profit']
    
    def train(self, episodes: Optional[int] = None):
        """
        Train agents in the market environment

        Args:
            episodes: Number of training episodes (default from experiment config or TRAINING_CONFIG)
        """
        if episodes is None:
            # Use experiment-specific training episodes if available, otherwise use global default
            if 'training' in self.experiment_config and 'episodes' in self.experiment_config['training']:
                episodes = self.experiment_config['training']['episodes']
            else:
                episodes = TRAINING_CONFIG['training_phase']['episodes']
        
        max_steps = TRAINING_CONFIG['training_phase']['max_steps_per_episode']
        print_interval = TRAINING_CONFIG['training_phase']['print_interval']
        save_interval = TRAINING_CONFIG['training_phase']['save_interval']
        
        if self.verbose:
            print(f"\nStarting training for {episodes} episodes...")
            print("=" * 70)
        
        for episode in range(episodes):
            # Reset environment
            states, _ = self.env.reset()
            
            # Episode tracking
            episode_rewards = np.zeros(self.n_firms)
            episode_prices = [[] for _ in range(self.n_firms)]
            episode_profits = [[] for _ in range(self.n_firms)]
            episode_shares = [[] for _ in range(self.n_firms)]
            
            for step in range(max_steps):
                # Get actions from all agents
                actions = []
                for i, agent in enumerate(self.agents):
                    action = agent.act(states[i], training=True)
                    actions.append(action)
                
                # Execute actions in environment
                next_states, rewards, terminated, truncated, info = self.env.step(np.array(actions))
                done = terminated or truncated
                
                # Store experiences and learn
                for i, agent in enumerate(self.agents):
                    agent.step(states[i], actions[i], rewards[i], next_states[i], done)
                
                # Track metrics
                episode_rewards += rewards
                for i in range(self.n_firms):
                    episode_prices[i].append(info['prices'][i])
                    episode_profits[i].append(info['profits'][i])
                    episode_shares[i].append(info['market_shares'][i])
                
                states = next_states
                
                if done:
                    break
            
            # Decay epsilon for all agents
            for agent in self.agents:
                agent.decay_epsilon()
            
            # Store episode statistics
            for i in range(self.n_firms):
                self.training_history['episode_rewards'][i].append(episode_rewards[i])
                self.training_history['episode_prices'][i].append(np.mean(episode_prices[i]))
                self.training_history['episode_profits'][i].append(np.mean(episode_profits[i]))
                self.training_history['episode_shares'][i].append(np.mean(episode_shares[i]))
            
            # Calculate and store metrics
            rpdi, delta = self._calculate_episode_metrics(episode_prices, episode_profits)
            self.training_history['rpdi'].append(rpdi)
            self.training_history['delta'].append(delta)
            
            # Print progress
            if (episode + 1) % print_interval == 0:
                self._print_training_progress(episode + 1)
            
            # Save checkpoints
            if (episode + 1) % save_interval == 0:
                self._save_checkpoint(episode + 1)
        
        if self.verbose:
            print("=" * 70)
            print("Training completed!")
    
    def evaluate(self, episodes: Optional[int] = None):
        """
        Evaluate trained agents without learning
        
        Args:
            episodes: Number of evaluation episodes (default from config)
        """
        if episodes is None:
            episodes = TRAINING_CONFIG['evaluation_phase']['episodes']
        
        max_steps = TRAINING_CONFIG['evaluation_phase']['max_steps_per_episode']
        metric_timesteps = TRAINING_CONFIG['evaluation_phase']['metric_timesteps']
        
        if self.verbose:
            print(f"\nEvaluating agents for {episodes} episodes...")
            print("=" * 70)
        
        # Collect all timestep data for metrics calculation
        all_prices = [[] for _ in range(self.n_firms)]
        all_profits = [[] for _ in range(self.n_firms)]
        all_shares = [[] for _ in range(self.n_firms)]
        
        for episode in range(episodes):
            states, _ = self.env.reset()
            
            for step in range(max_steps):
                # Get actions without exploration
                actions = []
                for i, agent in enumerate(self.agents):
                    action = agent.act(states[i], training=False)
                    actions.append(action)
                
                next_states, rewards, terminated, truncated, info = self.env.step(np.array(actions))
                
                # Collect data
                for i in range(self.n_firms):
                    all_prices[i].append(info['prices'][i])
                    all_profits[i].append(info['profits'][i])
                    all_shares[i].append(info['market_shares'][i])
                
                states = next_states
                
                if terminated or truncated:
                    break
        
        # Calculate metrics using last N timesteps
        self.evaluation_results = self._calculate_evaluation_metrics(
            all_prices, all_profits, all_shares, metric_timesteps
        )
        
        if self.verbose:
            self._print_evaluation_results()
        
        return self.evaluation_results
    
    def _calculate_episode_metrics(
        self, 
        episode_prices: List[List[float]], 
        episode_profits: List[List[float]]
    ) -> Tuple[float, float]:
        """
        Calculate RPDI and Delta metrics for an episode
        
        Args:
            episode_prices: Prices for each firm during episode
            episode_profits: Profits for each firm during episode
            
        Returns:
            Tuple of (RPDI, Delta) values
        """
        # Calculate average prices and profits
        avg_prices = [np.mean(prices) for prices in episode_prices]
        avg_profits = [np.mean(profits) for profits in episode_profits]
        
        # Overall averages
        avg_price = np.mean(avg_prices)
        avg_profit = np.mean(avg_profits)
        
        # Calculate RPDI
        if self.monopoly_price_avg != self.nash_price_avg:
            rpdi = (avg_price - self.nash_price_avg) / (self.monopoly_price_avg - self.nash_price_avg)
        else:
            rpdi = 0.0
        
        # Calculate Delta
        if self.monopoly_profit_avg != self.nash_profit_avg:
            delta = (avg_profit - self.nash_profit_avg) / (self.monopoly_profit_avg - self.nash_profit_avg)
        else:
            delta = 0.0
        
        return rpdi, delta
    
    def _calculate_evaluation_metrics(
        self,
        all_prices: List[List[float]],
        all_profits: List[List[float]],
        all_shares: List[List[float]],
        last_n_steps: int
    ) -> Dict:
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            all_prices: All prices from evaluation
            all_profits: All profits from evaluation
            all_shares: All market shares from evaluation
            last_n_steps: Number of last timesteps to use
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = {
            'individual_firms': [],
            'overall': {},
            'interpretation': '',
        }
        
        # Calculate metrics for each firm
        for i in range(self.n_firms):
            # Use last N timesteps
            firm_prices = all_prices[i][-last_n_steps:] if len(all_prices[i]) > last_n_steps else all_prices[i]
            firm_profits = all_profits[i][-last_n_steps:] if len(all_profits[i]) > last_n_steps else all_profits[i]
            firm_shares = all_shares[i][-last_n_steps:] if len(all_shares[i]) > last_n_steps else all_shares[i]
            
            avg_price = np.mean(firm_prices)
            avg_profit = np.mean(firm_profits)
            avg_share = np.mean(firm_shares)
            
            # Calculate firm-specific RPDI and Delta
            if self.monopoly_prices[i] != self.nash_prices[i]:
                firm_rpdi = (avg_price - self.nash_prices[i]) / (self.monopoly_prices[i] - self.nash_prices[i])
            else:
                firm_rpdi = 0.0
            
            if self.monopoly_profits[i] != self.nash_profits[i]:
                firm_delta = (avg_profit - self.nash_profits[i]) / (self.monopoly_profits[i] - self.nash_profits[i])
            else:
                firm_delta = 0.0
            
            results['individual_firms'].append({
                'firm_id': i,
                'rpdi': firm_rpdi,
                'delta': firm_delta,
                'avg_price': avg_price,
                'avg_profit': avg_profit,
                'avg_share': avg_share,
                'price_std': np.std(firm_prices),
                'profit_std': np.std(firm_profits),
            })
        
        # Calculate overall metrics
        all_prices_flat = []
        all_profits_flat = []
        for i in range(self.n_firms):
            prices = all_prices[i][-last_n_steps:] if len(all_prices[i]) > last_n_steps else all_prices[i]
            profits = all_profits[i][-last_n_steps:] if len(all_profits[i]) > last_n_steps else all_profits[i]
            all_prices_flat.extend(prices)
            all_profits_flat.extend(profits)
        
        overall_avg_price = np.mean(all_prices_flat)
        overall_avg_profit = np.mean(all_profits_flat)
        
        # Overall RPDI and Delta
        if self.monopoly_price_avg != self.nash_price_avg:
            overall_rpdi = (overall_avg_price - self.nash_price_avg) / (self.monopoly_price_avg - self.nash_price_avg)
        else:
            overall_rpdi = 0.0
        
        if self.monopoly_profit_avg != self.nash_profit_avg:
            overall_delta = (overall_avg_profit - self.nash_profit_avg) / (self.monopoly_profit_avg - self.nash_profit_avg)
        else:
            overall_delta = 0.0
        
        results['overall'] = {
            'rpdi': overall_rpdi,
            'delta': overall_delta,
            'avg_price': overall_avg_price,
            'avg_profit': overall_avg_profit,
            'price_std': np.std(all_prices_flat),
            'profit_std': np.std(all_profits_flat),
        }
        
        # Interpret results
        results['interpretation'] = interpret_metrics(overall_rpdi, overall_delta)
        
        return results
    
    def _print_training_progress(self, episode: int):
        """Print training progress"""
        if not self.verbose:
            return
        
        # Calculate recent averages (last 100 episodes)
        window = min(100, episode)
        
        avg_rewards = []
        avg_prices = []
        for i in range(self.n_firms):
            avg_rewards.append(np.mean(self.training_history['episode_rewards'][i][-window:]))
            avg_prices.append(np.mean(self.training_history['episode_prices'][i][-window:]))
        
        avg_rpdi = np.mean(self.training_history['rpdi'][-window:])
        avg_delta = np.mean(self.training_history['delta'][-window:])
        avg_epsilon = np.mean([agent.epsilon for agent in self.agents])
        
        print(f"Episode {episode:4d} | "
              f"Avg Reward: {np.mean(avg_rewards):7.4f} | "
              f"Avg Price: {np.mean(avg_prices):7.4f} | "
              f"RPDI: {avg_rpdi:6.3f} | "
              f"Δ: {avg_delta:6.3f} | "
              f"ε: {avg_epsilon:5.3f}")
    
    def _print_evaluation_results(self):
        """Print evaluation results"""
        if not self.verbose or self.evaluation_results is None:
            return
        
        results = self.evaluation_results
        
        print("\n" + "=" * 70)
        print(f"EVALUATION RESULTS: {self.experiment_name}")
        print("=" * 70)
        
        print("\nBenchmark Values:")
        print(f"  Nash Price (avg):      {self.nash_price_avg:.4f}")
        print(f"  Monopoly Price (avg):  {self.monopoly_price_avg:.4f}")
        print(f"  Nash Profit (avg):     {self.nash_profit_avg:.4f}")
        print(f"  Monopoly Profit (avg): {self.monopoly_profit_avg:.4f}")
        
        print("\nIndividual Firm Results:")
        print("-" * 70)
        for firm_result in results['individual_firms']:
            print(f"Firm {firm_result['firm_id']}:")
            print(f"  RPDI:         {firm_result['rpdi']:7.4f}")
            print(f"  Delta:        {firm_result['delta']:7.4f}")
            print(f"  Avg Price:    {firm_result['avg_price']:7.4f}")
            print(f"  Avg Profit:   {firm_result['avg_profit']:7.4f}")
            print(f"  Market Share: {firm_result['avg_share']:7.4f}")
        
        print("\n" + "-" * 70)
        print("Overall Market Metrics:")
        print(f"  RPDI:         {results['overall']['rpdi']:7.4f}")
        print(f"  Delta:        {results['overall']['delta']:7.4f}")
        print(f"  Avg Price:    {results['overall']['avg_price']:7.4f}")
        print(f"  Avg Profit:   {results['overall']['avg_profit']:7.4f}")
        
        print("\n" + "=" * 70)
        print("MARKET BEHAVIOR ASSESSMENT:")
        print(results['interpretation'])
        print("=" * 70)
    
    def _save_checkpoint(self, episode: int):
        """Save model checkpoints"""
        checkpoint_dir = os.path.join(self.save_dir, f"checkpoint_{episode}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for agent in self.agents:
            agent.save(os.path.join(checkpoint_dir, f"agent_{agent.agent_id}.pth"))
        
        # Save training history
        history_path = os.path.join(checkpoint_dir, "training_history.json")
        with open(history_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            history_to_save = {}
            for key, value in self.training_history.items():
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], list):
                        history_to_save[key] = [[float(x) for x in sublist] for sublist in value]
                    else:
                        history_to_save[key] = [float(x) for x in value]
                else:
                    history_to_save[key] = value
            json.dump(history_to_save, f, indent=2)
    
    def save_results(self):
        """Save final results and plots"""
        # Save evaluation results
        if self.evaluation_results:
            results_path = os.path.join(self.save_dir, "evaluation_results.json")
            with open(results_path, 'w') as f:
                json.dump(self.evaluation_results, f, indent=2)
        
        # Save configuration
        config_path = os.path.join(self.save_dir, "experiment_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.experiment_config, f, indent=2)
        
        # Save plots
        self.plot_training_history()
        
        if self.verbose:
            print(f"\nResults saved to: {self.save_dir}")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Moving average window
        window = TRAINING_CONFIG['analysis']['moving_avg_window']
        
        # Plot rewards
        ax = axes[0, 0]
        for i in range(self.n_firms):
            rewards = self.training_history['episode_rewards'][i]
            smoothed = self._moving_average(rewards, window)
            ax.plot(smoothed, label=f'Firm {i}', alpha=0.7)
        ax.set_title('Episode Rewards')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot prices
        ax = axes[0, 1]
        for i in range(self.n_firms):
            prices = self.training_history['episode_prices'][i]
            smoothed = self._moving_average(prices, window)
            ax.plot(smoothed, label=f'Firm {i}', alpha=0.7)
        ax.axhline(y=self.nash_price_avg, color='g', linestyle='--', label='Nash', alpha=0.5)
        ax.axhline(y=self.monopoly_price_avg, color='r', linestyle='--', label='Monopoly', alpha=0.5)
        ax.set_title('Average Prices')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot profits
        ax = axes[0, 2]
        for i in range(self.n_firms):
            profits = self.training_history['episode_profits'][i]
            smoothed = self._moving_average(profits, window)
            ax.plot(smoothed, label=f'Firm {i}', alpha=0.7)
        ax.axhline(y=self.nash_profit_avg, color='g', linestyle='--', label='Nash', alpha=0.5)
        ax.axhline(y=self.monopoly_profit_avg, color='r', linestyle='--', label='Monopoly', alpha=0.5)
        ax.set_title('Average Profits')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Profit')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot market shares
        ax = axes[1, 0]
        for i in range(self.n_firms):
            shares = self.training_history['episode_shares'][i]
            smoothed = self._moving_average(shares, window)
            ax.plot(smoothed, label=f'Firm {i}', alpha=0.7)
        ax.set_title('Market Shares')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Market Share')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot RPDI
        ax = axes[1, 1]
        rpdi = self.training_history['rpdi']
        smoothed_rpdi = self._moving_average(rpdi, window)
        ax.plot(smoothed_rpdi, color='blue', alpha=0.7)
        ax.axhline(y=0.3, color='g', linestyle='--', label='Competitive', alpha=0.5)
        ax.axhline(y=0.7, color='r', linestyle='--', label='Collusive', alpha=0.5)
        ax.set_title('RPDI (Price Index)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('RPDI')
        ax.set_ylim([-0.1, 1.1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot Delta
        ax = axes[1, 2]
        delta = self.training_history['delta']
        smoothed_delta = self._moving_average(delta, window)
        ax.plot(smoothed_delta, color='purple', alpha=0.7)
        ax.axhline(y=0.3, color='g', linestyle='--', label='Competitive', alpha=0.5)
        ax.axhline(y=0.7, color='r', linestyle='--', label='Collusive', alpha=0.5)
        ax.set_title('Δ (Profit Index)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Delta')
        ax.set_ylim([-0.1, 1.1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Training History: {self.experiment_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.save_dir, "training_history.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create detailed price and profit dynamics charts
        self.plot_price_profit_dynamics()
    
    def plot_price_profit_dynamics(self):
        """
        Create detailed line charts showing price and profit dynamics across all training episodes
        with Nash and Monopoly benchmarks as reference lines
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Moving average window for smoother visualization
        window = TRAINING_CONFIG['analysis']['moving_avg_window']
        episodes = np.arange(len(self.training_history['episode_prices'][0]))
        
        # ===== Chart 1: Price Dynamics Across Episodes =====
        ax1 = axes[0]
        
        # Plot each firm's price trajectory
        for i in range(self.n_firms):
            prices = self.training_history['episode_prices'][i]
            smoothed_prices = self._moving_average(prices, window)
            
            # Adjust x-axis for moving average
            if len(smoothed_prices) < len(episodes):
                x_data = episodes[window-1:]
            else:
                x_data = episodes[:len(smoothed_prices)]
            
            ax1.plot(x_data, smoothed_prices, label=f'Firm {i}', 
                    linewidth=2, alpha=0.8)
        
        # Add Nash equilibrium reference line
        ax1.axhline(y=self.nash_price_avg, color='green', linestyle='--', 
                   linewidth=2.5, label='Nash Equilibrium', alpha=0.7)
        
        # Add Monopoly reference line
        ax1.axhline(y=self.monopoly_price_avg, color='red', linestyle='--', 
                   linewidth=2.5, label='Monopoly', alpha=0.7)
        
        # Add shaded region between Nash and Monopoly
        ax1.fill_between(episodes, self.nash_price_avg, self.monopoly_price_avg,
                        color='yellow', alpha=0.1, label='Intermediate Zone')
        
        ax1.set_xlabel('Training Episode', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Price', fontsize=12, fontweight='bold')
        ax1.set_title('Price Dynamics Across Training Episodes', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='best', framealpha=0.9, fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle=':', linewidth=1)
        
        # Add annotation for key regions
        y_mid = (self.nash_price_avg + self.monopoly_price_avg) / 2
        ax1.text(len(episodes) * 0.02, self.nash_price_avg * 0.98, 
                'Competitive', fontsize=9, color='green', 
                style='italic', verticalalignment='top')
        ax1.text(len(episodes) * 0.02, self.monopoly_price_avg * 1.01, 
                'Collusive', fontsize=9, color='red', 
                style='italic', verticalalignment='bottom')
        
        # ===== Chart 2: Profit Dynamics Across Episodes =====
        ax2 = axes[1]
        
        # Plot each firm's profit trajectory
        for i in range(self.n_firms):
            profits = self.training_history['episode_profits'][i]
            smoothed_profits = self._moving_average(profits, window)
            
            # Adjust x-axis for moving average
            if len(smoothed_profits) < len(episodes):
                x_data = episodes[window-1:]
            else:
                x_data = episodes[:len(smoothed_profits)]
            
            ax2.plot(x_data, smoothed_profits, label=f'Firm {i}', 
                    linewidth=2, alpha=0.8)
        
        # Add Nash equilibrium profit reference line
        ax2.axhline(y=self.nash_profit_avg, color='green', linestyle='--', 
                   linewidth=2.5, label='Nash Profit', alpha=0.7)
        
        # Add Monopoly profit reference line
        ax2.axhline(y=self.monopoly_profit_avg, color='red', linestyle='--', 
                   linewidth=2.5, label='Monopoly Profit', alpha=0.7)
        
        # Add shaded region between Nash and Monopoly profits
        ax2.fill_between(episodes, self.nash_profit_avg, self.monopoly_profit_avg,
                        color='yellow', alpha=0.1, label='Intermediate Zone')
        
        ax2.set_xlabel('Training Episode', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Profit', fontsize=12, fontweight='bold')
        ax2.set_title('Profit Dynamics Across Training Episodes', 
                     fontsize=14, fontweight='bold')
        ax2.legend(loc='best', framealpha=0.9, fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle=':', linewidth=1)
        
        # Add annotation for key regions
        ax2.text(len(episodes) * 0.02, self.nash_profit_avg * 0.95, 
                'Competitive', fontsize=9, color='green', 
                style='italic', verticalalignment='top')
        ax2.text(len(episodes) * 0.02, self.monopoly_profit_avg * 1.02, 
                'Collusive', fontsize=9, color='red', 
                style='italic', verticalalignment='bottom')
        
        # Overall title
        plt.suptitle(f'Training Dynamics: {self.experiment_name}', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save the detailed dynamics plot
        dynamics_path = os.path.join(self.save_dir, "price_profit_dynamics.png")
        plt.savefig(dynamics_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"\n📊 Price and profit dynamics chart saved to: {dynamics_path}")
    
    def _moving_average(self, data: List[float], window: int) -> np.ndarray:
        """Calculate moving average for smoothing"""
        if len(data) < window:
            return np.array(data)
        return np.convolve(data, np.ones(window)/window, mode='valid')