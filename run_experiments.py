#!/usr/bin/env python3
"""
Main script to run all DQN market simulation experiments

This script executes all experiments as specified in the README.md:
1. Basic 2-firm symmetric simulation
2. Extended 3-firm symmetric simulation
3. Extended 4-firm symmetric simulation
4. Extended 4-firm asymmetric simulation

Usage:
    python run_experiments.py [--experiment EXP_NUM] [--learning-rate LR] [--episodes N]

Examples:
    python run_experiments.py                    # Run all experiments
    python run_experiments.py --experiment 1     # Run only 2-firm experiment
    python run_experiments.py --learning-rate 0.05 --episodes 1000
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import experiment configurations
from experiment_config import (
    EXPERIMENT_2FIRM_SYMMETRIC,
    EXPERIMENT_3FIRM_SYMMETRIC,
    EXPERIMENT_4FIRM_SYMMETRIC,
    EXPERIMENT_4FIRM_ASYMMETRIC,
    TRAINING_CONFIG,
    DQN_HYPERPARAMS
)

# Import simulation module
from market_simulation import MarketSimulation


def print_header(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80))
    print("=" * 80)


def run_experiment(config, learning_rate=None, episodes=None, verbose=True):
    """
    Run a single experiment
    
    Args:
        config: Experiment configuration dictionary
        learning_rate: Optional learning rate override
        episodes: Optional number of episodes override
        verbose: Whether to print detailed output
    
    Returns:
        Simulation object and evaluation results
    """
    experiment_name = config['name']
    
    if verbose:
        print_header(f"Running: {experiment_name}")
        print(f"\nConfiguration:")
        print(f"  • Firms: {config['market_structure']['n_firms']}")
        print(f"  • Substitutability: {config['market_parameters']['substitutability']}")
        print(f"  • Learning rate: {learning_rate or DQN_HYPERPARAMS['learning']['default_lr']}")
        print(f"  • Episodes: {episodes or TRAINING_CONFIG['training_phase']['episodes']}")
    
    # Create simulation
    sim = MarketSimulation(
        experiment_config=config,
        learning_rate=learning_rate,
        save_dir=f"results/{experiment_name.lower().replace(' ', '_').replace('-', '')}",
        verbose=verbose
    )
    
    # Train
    training_episodes = episodes or TRAINING_CONFIG['training_phase']['episodes']
    sim.train(episodes=training_episodes)
    
    # Evaluate
    eval_episodes = TRAINING_CONFIG['evaluation_phase']['episodes']
    eval_results = sim.evaluate(episodes=eval_episodes)
    
    # Save results
    sim.save_results()
    
    return sim, eval_results


def run_all_experiments(learning_rate=None, episodes=None):
    """Run all experiments and compile results"""
    
    print_header("DQN MARKET SIMULATION - COMPLETE EXPERIMENT SUITE")
    print("\nExperiments to run:")
    print("  1. 2-Firm Symmetric (Basic)")
    print("  2. 3-Firm Symmetric (Extended)")
    print("  3. 4-Firm Symmetric (Extended)")
    print("  4. 4-Firm Asymmetric (Extended)")
    
    # Store all results
    all_results = {}
    
    # Experiment 1: 2-Firm Symmetric
    print_header("EXPERIMENT 1: 2-FIRM SYMMETRIC")
    
    # Test multiple learning rates for 2-firm case
    if learning_rate is None:
        learning_rates_to_test = DQN_HYPERPARAMS['learning']['learning_rates']
        print(f"\nTesting learning rates: {learning_rates_to_test}")
    else:
        learning_rates_to_test = [learning_rate]
    
    exp1_results = {}
    for lr in learning_rates_to_test:
        print(f"\n→ Testing learning rate: {lr}")
        sim, eval_res = run_experiment(
            EXPERIMENT_2FIRM_SYMMETRIC, 
            learning_rate=lr, 
            episodes=episodes,
            verbose=True
        )
        exp1_results[lr] = {
            'simulation': sim,
            'evaluation': eval_res
        }
    
    # Select best learning rate based on RPDI closest to 0.5 (balanced)
    best_lr = min(exp1_results.keys(), 
                  key=lambda x: abs(exp1_results[x]['evaluation']['overall']['rpdi'] - 0.5))
    print(f"\n✅ Best learning rate selected: {best_lr}")
    all_results['2-Firm Symmetric'] = exp1_results[best_lr]
    
    # Experiment 2: 3-Firm Symmetric
    print_header("EXPERIMENT 2: 3-FIRM SYMMETRIC")
    sim, eval_res = run_experiment(
        EXPERIMENT_3FIRM_SYMMETRIC,
        learning_rate=best_lr,
        episodes=episodes,
        verbose=True
    )
    all_results['3-Firm Symmetric'] = {
        'simulation': sim,
        'evaluation': eval_res
    }
    
    # Experiment 3: 4-Firm Symmetric
    print_header("EXPERIMENT 3: 4-FIRM SYMMETRIC")
    sim, eval_res = run_experiment(
        EXPERIMENT_4FIRM_SYMMETRIC,
        learning_rate=best_lr,
        episodes=episodes,
        verbose=True
    )
    all_results['4-Firm Symmetric'] = {
        'simulation': sim,
        'evaluation': eval_res
    }
    
    # Experiment 4: 4-Firm Asymmetric
    print_header("EXPERIMENT 4: 4-FIRM ASYMMETRIC")
    sim, eval_res = run_experiment(
        EXPERIMENT_4FIRM_ASYMMETRIC,
        learning_rate=best_lr,
        episodes=episodes,
        verbose=True
    )
    all_results['4-Firm Asymmetric'] = {
        'simulation': sim,
        'evaluation': eval_res
    }
    
    return all_results, best_lr


def compile_and_display_results(all_results):
    """Compile and display comparative results"""
    
    print_header("COMPARATIVE RESULTS SUMMARY")
    
    # Create summary table
    summary_data = []
    for scenario, results in all_results.items():
        eval_data = results['evaluation']
        summary_data.append({
            'Scenario': scenario,
            'RPDI': eval_data['overall']['rpdi'],
            'Delta': eval_data['overall']['delta'],
            'Avg Price': eval_data['overall']['avg_price'],
            'Avg Profit': eval_data['overall']['avg_profit'],
            'Behavior': eval_data['interpretation'].split(':')[0].strip()
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Format for display
    df_display = df_summary.copy()
    df_display['RPDI'] = df_display['RPDI'].map('{:.4f}'.format)
    df_display['Delta'] = df_display['Delta'].map('{:.4f}'.format)
    df_display['Avg Price'] = df_display['Avg Price'].map('{:.3f}'.format)
    df_display['Avg Profit'] = df_display['Avg Profit'].map('{:.3f}'.format)
    
    print("\n" + df_display.to_string(index=False))
    
    # Save summary to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"results/summary_{timestamp}.csv"
    df_summary.to_csv(summary_file, index=False)
    print(f"\n✅ Summary saved to: {summary_file}")
    
    # Print key insights
    print_header("KEY INSIGHTS")
    
    # Analyze market structure effect
    print("\n📊 Effect of Market Structure (Symmetric Cases):")
    symmetric_data = df_summary[df_summary['Scenario'].str.contains('Symmetric')]
    symmetric_data = symmetric_data.sort_values('Scenario')
    
    rpdi_values = symmetric_data['RPDI'].values
    if len(rpdi_values) >= 3:
        if rpdi_values[0] > rpdi_values[1] > rpdi_values[2]:
            print("  → More firms lead to MORE COMPETITIVE pricing")
        elif rpdi_values[0] < rpdi_values[1] < rpdi_values[2]:
            print("  → More firms lead to MORE COLLUSIVE pricing")
        else:
            print("  → Non-monotonic relationship between firm count and pricing")
    
    # Compare symmetric vs asymmetric
    if '4-Firm Symmetric' in all_results and '4-Firm Asymmetric' in all_results:
        print("\n🔄 Symmetric vs Asymmetric (4-Firm Case):")
        sym_rpdi = df_summary[df_summary['Scenario'] == '4-Firm Symmetric']['RPDI'].values[0]
        asym_rpdi = df_summary[df_summary['Scenario'] == '4-Firm Asymmetric']['RPDI'].values[0]
        
        if asym_rpdi > sym_rpdi:
            print(f"  → Asymmetry increases RPDI by {(asym_rpdi - sym_rpdi):.4f}")
            print("  → Heterogeneity facilitates collusion")
        else:
            print(f"  → Asymmetry decreases RPDI by {(sym_rpdi - asym_rpdi):.4f}")
            print("  → Heterogeneity intensifies competition")
    
    # Market behavior distribution
    print("\n🎯 Market Behavior Distribution:")
    behavior_counts = df_summary['Behavior'].value_counts()
    for behavior, count in behavior_counts.items():
        percentage = (count / len(df_summary)) * 100
        print(f"  • {behavior}: {count} scenarios ({percentage:.0f}%)")
    
    return df_summary


def main():
    """Main execution function"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run DQN market simulation experiments'
    )
    parser.add_argument(
        '--experiment',
        type=int,
        choices=[1, 2, 3, 4],
        help='Run specific experiment (1: 2-firm, 2: 3-firm, 3: 4-firm symmetric, 4: 4-firm asymmetric)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        help='Override default learning rate'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        help='Override default number of training episodes'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test with fewer episodes (500)'
    )
    
    args = parser.parse_args()
    
    # Set episodes for quick test
    if args.quick:
        episodes = 500
        print("⚡ Running in QUICK MODE (500 episodes)")
    else:
        episodes = args.episodes
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    try:
        if args.experiment:
            # Run specific experiment
            configs = {
                1: EXPERIMENT_2FIRM_SYMMETRIC,
                2: EXPERIMENT_3FIRM_SYMMETRIC,
                3: EXPERIMENT_4FIRM_SYMMETRIC,
                4: EXPERIMENT_4FIRM_ASYMMETRIC
            }
            
            config = configs[args.experiment]
            print_header(f"Running Experiment {args.experiment}: {config['name']}")
            
            sim, eval_results = run_experiment(
                config,
                learning_rate=args.learning_rate,
                episodes=episodes,
                verbose=True
            )
            
            # Display results
            print_header("RESULTS")
            print(f"\nRPDI: {eval_results['overall']['rpdi']:.4f}")
            print(f"Delta: {eval_results['overall']['delta']:.4f}")
            print(f"Interpretation: {eval_results['interpretation']}")
            
        else:
            # Run all experiments
            all_results, best_lr = run_all_experiments(
                learning_rate=args.learning_rate,
                episodes=episodes
            )
            
            # Compile and display results
            df_summary = compile_and_display_results(all_results)
        
        print("\n" + "=" * 80)
        print("✅ EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Experiments interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()