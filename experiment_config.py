"""
Experiment Configuration for DQN Market Simulation

This file contains all experimental parameter settings based on the README.md
experiment design specification. It includes configurations for:
- Experiment 1: Basic 2-firm symmetric simulation
- Experiment 2: Extended 3-firm symmetric simulation
- Experiment 3: Extended 4-firm symmetric simulation
- Experiment 4: Extended 3-firm asymmetric simulation
- Experiment 5: Extended 4-firm asymmetric simulation
"""

import numpy as np

# ===============================================================================
# DQN Hyperparameters (Common across all experiments)
# ===============================================================================

DQN_HYPERPARAMS = {
    'network_architecture': {
        'hidden_layers': 2,
        'hidden_size': 64,
        'activation': 'relu',
    },
    'learning': {
        'learning_rates': [0.01, 0.05, 0.1],  # Test multiple learning rates
        'default_lr': 0.01,
        'optimizer': 'adam',
        'gamma': 0.98,  # Discount factor (increased from 0.95 to promote collusion)
        'tau': 1e-3,    # Soft update parameter
    },
    'exploration': {
        'epsilon_start': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.998,  # Slowed down from 0.995 to allow more exploration
    },
    'experience_replay': {
        'buffer_size': 100000,
        'batch_size': 32,
        'update_frequency': 4,
    }
}

# ===============================================================================
# Training Configuration (Common across all experiments)
# ===============================================================================

TRAINING_CONFIG = {
    'training_phase': {
        'episodes': 2000,  # Total training episodes
        'max_steps_per_episode': 1000,
        'save_interval': 100,
        'print_interval': 100,
    },
    'evaluation_phase': {
        'episodes': 100,  # Post-training evaluation episodes
        'max_steps_per_episode': 1000,
        'metric_timesteps': 10000,  # Last 10,000 timesteps for metric calculation
    },
    'analysis': {
        'moving_avg_window': 50,
        'n_repetitions': 20,  # Number of independent runs for robustness
    }
}

# ===============================================================================
# Experiment 1: Basic 2-Agent Symmetric Simulation
# ===============================================================================

EXPERIMENT_2FIRM_SYMMETRIC = {
    'name': '2-Firm Symmetric',
    'description': 'Basic DQN simulation with 2 symmetric firms',

    'market_structure': {
        'n_firms': 2,
        'n_actions': 15,  # Discrete price actions
    },

    'firm_parameters': {
        'marginal_costs': [1.0, 1.0],  # c = 1 for both firms
        'product_qualities': [2.0, 2.0],  # g = 2 for both firms
    },

    'market_parameters': {
        'substitutability': 0.4,  # μ = 0.4
        'outside_option': 0.0,  # Normalized to 0
    },

    'benchmarks': {
        'nash_price': 1.677,
        'monopoly_price': 2.071,
        'nash_profit': 0.277,
        'monopoly_profit': 0.335,
    },

    'price_range': {
        'xi': 0.1,  # Price range parameter ξ
        'min_price': 1.637,  # p_N - ξ(p_M - p_N)
        'max_price': 2.110,  # p_M + ξ(p_M - p_N)
    },

    'training': {
        'episodes': 2000,  # Basic simulation uses 2000 episodes
    },
}

# ===============================================================================
# Experiment 2: Extended 3-Agent Symmetric Simulation
# ===============================================================================

EXPERIMENT_3FIRM_SYMMETRIC = {
    'name': '3-Firm Symmetric',
    'description': 'Extended DQN simulation with 3 symmetric firms',

    'market_structure': {
        'n_firms': 3,
        'n_actions': 15,
    },

    'firm_parameters': {
        'marginal_costs': [1.0, 1.0, 1.0],
        'product_qualities': [2.0, 2.0, 2.0],
    },

    'market_parameters': {
        'substitutability': 0.4,  # μ = 0.4
        'outside_option': 0.0,
    },

    'benchmarks': {
        'nash_price': 1.521,
        'monopoly_price': 2.252,
        'nash_profit': 0.121,
        'monopoly_profit': 0.213,
    },

    'price_range': {
        'xi': 0.1,
        'min_price': 1.510,  # Common range for 3-firm scenarios
        'max_price': 2.235,  # Common range for 3-firm scenarios
    },

    'training': {
        'episodes': 1000,  # Extended simulation uses 1000 episodes
    },
}

# ===============================================================================
# Experiment 3: Extended 4-Agent Symmetric Simulation
# ===============================================================================

EXPERIMENT_4FIRM_SYMMETRIC = {
    'name': '4-Firm Symmetric',
    'description': 'Extended DQN simulation with 4 symmetric firms',

    'market_structure': {
        'n_firms': 4,
        'n_actions': 15,
    },

    'firm_parameters': {
        'marginal_costs': [1.0, 1.0, 1.0, 1.0],
        'product_qualities': [2.0, 2.0, 2.0, 2.0],
    },

    'market_parameters': {
        'substitutability': 0.4,  # μ = 0.4
        'outside_option': 0.0,
    },

    'benchmarks': {
        'nash_price': 1.521,
        'monopoly_price': 2.252,
        'nash_profit': 0.121,
        'monopoly_profit': 0.213,
    },

    'price_range': {
        'xi': 0.1,
        'min_price': 1.448,  # Common range for 4-firm scenarios
        'max_price': 2.325,  # Common range for 4-firm scenarios
    },

    'training': {
        'episodes': 1000,  # Extended simulation uses 1000 episodes
    },
}

# ===============================================================================
# Experiment 4: Extended 3-Agent Asymmetric Simulation
# ===============================================================================

EXPERIMENT_3FIRM_ASYMMETRIC = {
    'name': '3-Firm Asymmetric',
    'description': 'Extended DQN simulation with 3 firms having asymmetric parameters',

    'market_structure': {
        'n_firms': 3,
        'n_actions': 15,
    },

    'firm_parameters': {
        'marginal_costs': [1.1, 1.00, 0.95],
        'product_qualities': [2.08, 2.02, 1.96],
    },

    'market_parameters': {
        'substitutability': 0.4,  # μ = 0.4
        'outside_option': 0.0,
    },

    'benchmarks': {
        # Per-firm Nash and Monopoly prices
        'nash_prices': [1.662, 1.577, 1.524],
        'monopoly_prices': [2.277, 2.177, 2.127],
        'nash_profits': [0.162, 0.177, 0.174],
        'monopoly_profits': [0.244, 0.270, 0.263],

        # Average values for overall metrics
        'nash_price_avg': np.mean([1.662, 1.577, 1.524]),
        'monopoly_price_avg': np.mean([2.277, 2.177, 2.127]),
        'nash_profit_avg': np.mean([0.162, 0.177, 0.174]),
        'monopoly_profit_avg': np.mean([0.244, 0.270, 0.263]),
    },

    'price_range': {
        'xi': 0.1,
        # Per-firm price ranges (for reference)
        'min_prices': [1.600, 1.517, 1.463],
        'max_prices': [2.339, 2.237, 2.187],

        # Overall range (for common action space)
        'min_price': 1.510,  # Common range for 3-firm scenarios
        'max_price': 2.235,  # Common range for 3-firm scenarios
    },

    'training': {
        'episodes': 1000,  # Extended simulation uses 1000 episodes
    },
}

# ===============================================================================
# Experiment 5: Extended 4-Agent Asymmetric Simulation
# ===============================================================================

EXPERIMENT_4FIRM_ASYMMETRIC = {
    'name': '4-Firm Asymmetric',
    'description': 'Extended DQN simulation with 4 firms having asymmetric parameters',

    'market_structure': {
        'n_firms': 4,
        'n_actions': 15,
    },

    'firm_parameters': {
        'marginal_costs': [1.1, 1.00, 0.95, 0.90],
        'product_qualities': [2.08, 2.02, 1.96, 1.90],
    },

    'market_parameters': {
        'substitutability': 0.4,  # μ = 0.4
        'outside_option': 0.0,
    },

    'benchmarks': {
        # Per-firm Nash and Monopoly prices
        'nash_prices': [1.615, 1.526, 1.473, 1.420],
        'monopoly_prices': [2.354, 2.254, 2.204, 2.154],
        'nash_profits': [0.115, 0.126, 0.123, 0.120],
        'monopoly_profits': [0.202, 0.223, 0.217, 0.212],

        # Average values for overall metrics
        'nash_price_avg': np.mean([1.615, 1.526, 1.473, 1.420]),
        'monopoly_price_avg': np.mean([2.354, 2.254, 2.204, 2.154]),
        'nash_profit_avg': np.mean([0.115, 0.126, 0.123, 0.120]),
        'monopoly_profit_avg': np.mean([0.202, 0.223, 0.217, 0.212]),
    },

    'price_range': {
        'xi': 0.1,
        # Per-firm price ranges (for reference)
        'min_prices': [1.541, 1.453, 1.400, 1.346],
        'max_prices': [2.428, 2.327, 2.277, 2.227],

        # Overall range (for common action space)
        'min_price': 1.448,  # Common range for 4-firm scenarios
        'max_price': 2.325,  # Common range for 4-firm scenarios
    },

    'training': {
        'episodes': 1000,  # Extended simulation uses 1000 episodes
    },
}

# ===============================================================================
# Evaluation Metrics Configuration
# ===============================================================================

EVALUATION_METRICS = {
    'rpdi': {
        'name': 'Relative Price Deviation Index',
        'formula': '(p̄ - p^N) / (p^M - p^N)',
        'interpretation': {
            'competitive': (0.0, 0.3),  # RPDI < 0.3
            'intermediate': (0.3, 0.7),  # 0.3 ≤ RPDI ≤ 0.7
            'collusive': (0.7, 1.0),    # RPDI > 0.7
        }
    },
    'delta': {
        'name': 'Profit Metric',
        'formula': '(π̄ - π^N) / (π^M - π^N)',
        'interpretation': {
            'competitive': (0.0, 0.3),  # Δ < 0.3
            'intermediate': (0.3, 0.7),  # 0.3 ≤ Δ ≤ 0.7
            'collusive': (0.7, 1.0),    # Δ > 0.7
        }
    }
}

# ===============================================================================
# Experiment List (for batch processing)
# ===============================================================================

ALL_EXPERIMENTS = [
    EXPERIMENT_2FIRM_SYMMETRIC,
    EXPERIMENT_3FIRM_SYMMETRIC,
    EXPERIMENT_4FIRM_SYMMETRIC,
    EXPERIMENT_3FIRM_ASYMMETRIC,
    EXPERIMENT_4FIRM_ASYMMETRIC,
]

# ===============================================================================
# Helper Functions
# ===============================================================================

def get_experiment_config(experiment_name):
    """
    Get configuration for a specific experiment by name
    
    Args:
        experiment_name: Name of the experiment (e.g., '2-Firm Symmetric')
    
    Returns:
        Experiment configuration dictionary
    """
    for exp in ALL_EXPERIMENTS:
        if exp['name'] == experiment_name:
            return exp
    raise ValueError(f"Experiment '{experiment_name}' not found")

def calculate_price_range(nash_price, monopoly_price, xi=0.1):
    """
    Calculate price range for discrete action space
    
    Args:
        nash_price: Nash equilibrium price p^N
        monopoly_price: Monopoly price p^M
        xi: Range parameter (default: 0.1)
    
    Returns:
        Tuple of (min_price, max_price)
    """
    price_diff = monopoly_price - nash_price
    min_price = nash_price - xi * price_diff
    max_price = monopoly_price + xi * price_diff
    return min_price, max_price

def get_dqn_config_for_experiment(experiment_config, learning_rate=None):
    """
    Get DQN configuration merged with experiment-specific settings
    
    Args:
        experiment_config: Experiment configuration dictionary
        learning_rate: Optional specific learning rate (default: use default_lr)
    
    Returns:
        Complete DQN configuration for the experiment
    """
    dqn_config = {
        'learning_rate': learning_rate or DQN_HYPERPARAMS['learning']['default_lr'],
        'epsilon_start': DQN_HYPERPARAMS['exploration']['epsilon_start'],
        'epsilon_min': DQN_HYPERPARAMS['exploration']['epsilon_min'],
        'epsilon_decay': DQN_HYPERPARAMS['exploration']['epsilon_decay'],
        'gamma': DQN_HYPERPARAMS['learning']['gamma'],
        'tau': DQN_HYPERPARAMS['learning']['tau'],
        'buffer_size': DQN_HYPERPARAMS['experience_replay']['buffer_size'],
        'batch_size': DQN_HYPERPARAMS['experience_replay']['batch_size'],
        'hidden_size': DQN_HYPERPARAMS['network_architecture']['hidden_size'],
        'update_frequency': DQN_HYPERPARAMS['experience_replay']['update_frequency'],
    }
    return dqn_config

def interpret_metrics(rpdi, delta):
    """
    Interpret RPDI and Delta metrics to determine market behavior
    
    Args:
        rpdi: Relative Price Deviation Index value
        delta: Profit Metric value
    
    Returns:
        String describing the market behavior
    """
    if rpdi < 0.3 and delta < 0.3:
        return "✅ COMPETITIVE: Prices near Nash equilibrium, market operates efficiently"
    elif rpdi > 0.7 and delta > 0.7:
        return "🚨 COLLUSIVE: Prices near monopoly level, evidence of algorithmic collusion"
    else:
        return "⚠️ INTERMEDIATE: Partial coordination, moderate market power"