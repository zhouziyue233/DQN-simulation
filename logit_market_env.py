"""
Logit Bertrand Market Environment for DQN Simulation

This module implements the Logit Bertrand competition model as described in the README.
The environment supports both symmetric and asymmetric firm configurations.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, List, Optional


class LogitBertrandMarket(gym.Env):
    """
    Logit Bertrand market environment for multi-agent price competition
    
    The demand function follows the logit model:
    d_i = exp((g_i - p_i) / μ) / [exp(0) + Σ_j exp((g_j - p_j) / μ)]
    
    where:
    - d_i: firm i's demand (market share)
    - g_i: firm i's product quality
    - p_i: firm i's price
    - μ: substitutability parameter (lower = more substitutable)
    - exp(0) = 1: outside option (consumers can choose not to buy)
    """
    
    def __init__(
        self,
        n_firms: int,
        n_actions: int,
        marginal_costs: List[float],
        product_qualities: List[float],
        substitutability: float,
        price_range: Tuple[float, float],
        memory_length: int = 10,
        max_steps: int = 1000,
    ):
        """
        Initialize the Logit Bertrand market environment
        
        Args:
            n_firms: Number of competing firms
            n_actions: Number of discrete price actions
            marginal_costs: List of marginal costs for each firm
            product_qualities: List of product qualities for each firm
            substitutability: μ parameter (product substitutability)
            price_range: Tuple of (min_price, max_price) for action space
            memory_length: Length of price/profit history in state
            max_steps: Maximum steps per episode
        """
        super(LogitBertrandMarket, self).__init__()
        
        self.n_firms = n_firms
        self.n_actions = n_actions
        self.memory_length = memory_length
        self.max_steps = max_steps
        
        # Firm-specific parameters
        self.marginal_costs = np.array(marginal_costs, dtype=np.float32)
        self.product_qualities = np.array(product_qualities, dtype=np.float32)
        
        # Market parameters
        self.substitutability = substitutability
        self.min_price = price_range[0]
        self.max_price = price_range[1]
        self.price_range = self.max_price - self.min_price
        
        # Create discrete price grid
        self.price_grid = np.linspace(self.min_price, self.max_price, n_actions)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(n_actions)
        
        # State includes: own price history, own profit history, 
        # average competitor price, market stats
        state_size = 2 * memory_length + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(state_size,), dtype=np.float32
        )
        
        # Initialize tracking variables
        self.reset()
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state
        
        Returns:
            Initial observations for all agents and info dict
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 0
        
        # Initialize history arrays
        self.price_history = np.ones((self.n_firms, self.memory_length)) * np.mean([self.min_price, self.max_price])
        self.profit_history = np.zeros((self.n_firms, self.memory_length))
        self.market_shares = np.ones(self.n_firms) / (self.n_firms + 1)  # Initial equal shares
        
        # Initialize with random prices in the first timestep
        initial_actions = np.random.randint(0, self.n_actions, self.n_firms)
        initial_prices = self.price_grid[initial_actions]
        self.price_history[:, -1] = initial_prices
        
        # Calculate initial profits
        shares, profits = self._calculate_demand_and_profit(initial_prices)
        self.profit_history[:, -1] = profits
        self.market_shares = shares
        
        states = self._get_observations()
        info = {'initial_prices': initial_prices, 'initial_profits': profits}
        
        return states, info
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            actions: Array of action indices for each agent
            
        Returns:
            observations: New state observations for all agents
            rewards: Rewards (profits) for each agent
            terminated: Whether episode has ended
            truncated: Whether episode was truncated
            info: Additional information dictionary
        """
        # Convert actions to prices
        prices = self.price_grid[actions]
        
        # Calculate market shares and profits using Logit model
        shares, profits = self._calculate_demand_and_profit(prices)
        
        # Update history (shift old values left, add new values at the end)
        self.price_history = np.roll(self.price_history, -1, axis=1)
        self.profit_history = np.roll(self.profit_history, -1, axis=1)
        self.price_history[:, -1] = prices
        self.profit_history[:, -1] = profits
        self.market_shares = shares
        
        # Get new observations
        observations = self._get_observations()
        
        # Rewards are the profits
        rewards = profits
        
        # Check if episode is done
        self.current_step += 1
        terminated = False  # Episodes don't terminate naturally
        truncated = self.current_step >= self.max_steps
        
        # Compile info dictionary
        info = {
            'prices': prices,
            'profits': profits,
            'market_shares': shares,
            'step': self.current_step,
        }
        
        return observations, rewards, terminated, truncated, info
    
    def _calculate_demand_and_profit(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate market shares and profits using Logit demand model
        
        Args:
            prices: Array of prices for each firm
            
        Returns:
            market_shares: Demand/market share for each firm
            profits: Profit for each firm
        """
        # Calculate utilities: (g_i - p_i) / μ
        utilities = (self.product_qualities - prices) / self.substitutability
        
        # Prevent numerical overflow in exp()
        max_util = np.max(utilities)
        max_util = max(0.0, max_util)  # Compare with outside option (0)
        
        # Calculate exponentials with numerical stability
        exp_utilities = np.exp(utilities - max_util)
        exp_outside = np.exp(-max_util)  # Outside option utility = 0
        
        # Calculate market shares using logit formula
        denominator = exp_outside + np.sum(exp_utilities)
        market_shares = exp_utilities / denominator
        
        # Calculate profits: π_i = d_i × (p_i - c_i)
        profits = market_shares * (prices - self.marginal_costs)
        
        return market_shares, profits
    
    def _get_observations(self) -> np.ndarray:
        """
        Get current state observations for all agents
        
        State includes:
        - Agent's own price history
        - Agent's own profit history
        - Average competitor price
        - Price variance
        - Market share
        - Profit margin
        
        Returns:
            Array of observations for each agent
        """
        observations = []
        
        for i in range(self.n_firms):
            # Own price history (normalized)
            own_prices = (self.price_history[i] - self.min_price) / self.price_range
            
            # Own profit history (clipped for stability)
            own_profits = np.clip(self.profit_history[i], -1, 2)
            
            # Average competitor price (normalized)
            competitor_prices = np.delete(self.price_history[:, -1], i)
            avg_competitor_price = np.mean(competitor_prices) if len(competitor_prices) > 0 else self.price_history[i, -1]
            avg_competitor_price_norm = (avg_competitor_price - self.min_price) / self.price_range
            
            # Price variance in the market
            price_variance = np.var(self.price_history[:, -1])
            
            # Own market share
            own_share = self.market_shares[i]
            
            # Profit margin (price - cost) / price
            current_price = self.price_history[i, -1]
            profit_margin = (current_price - self.marginal_costs[i]) / current_price if current_price > 0 else 0
            
            # Combine all features
            observation = np.concatenate([
                own_prices,  # memory_length features
                own_profits,  # memory_length features
                [avg_competitor_price_norm, price_variance, own_share, profit_margin]  # 4 features
            ])
            
            observations.append(observation)
        
        return np.array(observations, dtype=np.float32)
    
    def get_info(self) -> Dict:
        """
        Get current environment information
        
        Returns:
            Dictionary with current market state information
        """
        return {
            'current_prices': self.price_history[:, -1],
            'current_profits': self.profit_history[:, -1],
            'market_shares': self.market_shares,
            'marginal_costs': self.marginal_costs,
            'product_qualities': self.product_qualities,
            'substitutability': self.substitutability,
            'step': self.current_step,
        }
    
    def render(self, mode='human'):
        """
        Render the environment (print current state)
        """
        if mode == 'human':
            print(f"\n--- Step {self.current_step} ---")
            for i in range(self.n_firms):
                print(f"Firm {i}: Price={self.price_history[i, -1]:.3f}, "
                      f"Profit={self.profit_history[i, -1]:.3f}, "
                      f"Share={self.market_shares[i]:.3f}")
            print(f"Avg Price: {np.mean(self.price_history[:, -1]):.3f}")
            print(f"Total Market Share: {np.sum(self.market_shares):.3f}")


def create_market_from_config(experiment_config: Dict) -> LogitBertrandMarket:
    """
    Create a LogitBertrandMarket environment from experiment configuration
    
    Args:
        experiment_config: Experiment configuration dictionary
        
    Returns:
        Configured LogitBertrandMarket environment
    """
    # Handle both symmetric and asymmetric cases
    if 'price_range' in experiment_config:
        if 'min_prices' in experiment_config['price_range']:
            # Asymmetric case - use overall range
            price_range = (
                experiment_config['price_range']['min_price'],
                experiment_config['price_range']['max_price']
            )
        else:
            # Symmetric case
            price_range = (
                experiment_config['price_range']['min_price'],
                experiment_config['price_range']['max_price']
            )
    else:
        raise ValueError("Price range not specified in experiment config")
    
    return LogitBertrandMarket(
        n_firms=experiment_config['market_structure']['n_firms'],
        n_actions=experiment_config['market_structure']['n_actions'],
        marginal_costs=experiment_config['firm_parameters']['marginal_costs'],
        product_qualities=experiment_config['firm_parameters']['product_qualities'],
        substitutability=experiment_config['market_parameters']['substitutability'],
        price_range=price_range,
        memory_length=10,
        max_steps=1000,
    )