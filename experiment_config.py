"""
Experiment Configuration for DQN Market Simulation

This file contains all experimental parameter settings based on the README.md
experiment design specification. It includes configurations for:
- Basic 2-agent symmetric simulation
- Extended 3-agent symmetric simulation  
- Extended 4-agent symmetric simulation
- Extended 4-agent asymmetric simulation
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
        'gamma': 0.95,  # Discount factor
        'tau': 1e-3,    # Soft update parameter
    },
    'exploration': {
        'epsilon_start': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
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
        'substitutability': 0.25,  # μ = 0.25
        'outside_option': 0.0,  # Normalized to 0
    },
    
    'benchmarks': {
        'nash_price': 1.473,
        'monopoly_price': 1.925,
        'nash_profit': 0.223,
        'monopoly_profit': 0.337,
    },
    
    'price_range': {
        'xi': 0.1,  # Price range parameter ξ
        'min_price': 1.428,  # p_N - ξ(p_M - p_N)
        'max_price': 1.970,  # p_M + ξ(p_M - p_N)
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
        'substitutability': 0.25,
        'outside_option': 0.0,
    },
    
    'benchmarks': {
        'nash_price': 1.370,
        'monopoly_price': 2.000,
        'nash_profit': 0.120,
        'monopoly_profit': 0.250,
    },
    
    'price_range': {
        'xi': 0.1,
        'min_price': 1.307,  # p_N - ξ(p_M - p_N)
        'max_price': 2.063,  # p_M + ξ(p_M - p_N)
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
        'substitutability': 0.25,
        'outside_option': 0.0,
    },
    
    'benchmarks': {
        'nash_price': 1.331,
        'monopoly_price': 2.054,
        'nash_profit': 0.081,
        'monopoly_profit': 0.201,
    },
    
    'price_range': {
        'xi': 0.1,
        'min_price': 1.259,  # p_N - ξ(p_M - p_N)
        'max_price': 2.126,  # p_M + ξ(p_M - p_N)
    },
}

# ===============================================================================
# Experiment 4: Extended 4-Agent Asymmetric Simulation
# ===============================================================================

EXPERIMENT_4FIRM_ASYMMETRIC = {
    'name': '4-Firm Asymmetric',
    'description': 'Extended DQN simulation with 4 firms having asymmetric parameters',
    
    'market_structure': {
        'n_firms': 4,
        'n_actions': 15,
    },
    
    'firm_parameters': {
        'marginal_costs': [1.05, 1.10, 0.95, 1.00],
        'product_qualities': [2.10, 2.00, 1.90, 1.80],
    },
    
    'market_parameters': {
        'substitutability': 0.30,  # μ = 0.30 for asymmetric case
        'outside_option': 0.0,
    },
    
    'benchmarks': {
        # Per-firm Nash and Monopoly prices
        'nash_prices': [1.486, 1.486, 1.351, 1.363],
        'monopoly_prices': [2.121, 2.171, 2.021, 2.071],
        'nash_profits': [0.136, 0.086, 0.101, 0.063],
        'monopoly_profits': [0.280, 0.170, 0.200, 0.122],
        
        # Average values for overall metrics
        'nash_price_avg': np.mean([1.486, 1.486, 1.351, 1.363]),
        'monopoly_price_avg': np.mean([2.121, 2.171, 2.021, 2.071]),
        'nash_profit_avg': np.mean([0.136, 0.086, 0.101, 0.063]),
        'monopoly_profit_avg': np.mean([0.280, 0.170, 0.200, 0.122]),
    },
    
    'price_range': {
        'xi': 0.1,
        # Per-firm price ranges
        'min_prices': [1.423, 1.418, 1.284, 1.292],
        'max_prices': [2.185, 2.240, 2.088, 2.142],
        
        # Overall range (for common action space)
        'min_price': 1.284,  # Min of all minimum prices
        'max_price': 2.240,  # Max of all maximum prices
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