# Algorithmic Pricing and Collusion: DQN Market Simulation

This is a Deep Q-Network (DQN) implementation for studying algorithmic collusion within oligopolistic markets.

## 💡 Academic Background

- As AI increasingly powers pricing algorithms in digital markets, a critical question emerges: **Can pricing algorithms learn to collude spontaneously?**

- Specifically, this question asks when pricing agents interact repeatedly in market, whether they could coordinate on higher prices for maxmizing long-term profits without huamn intervention in this process?

- In a computational experiment, Calvano et al (2020) have found that basic reinforcement learning (RL) algorithms like Q-learning are able to independently learn collusive pricing strategies and achieve supra-competitive profits even without explicit communication.

- However, Q-learning algorithm is primitive in reinforcement learning. It suffers from some technical limitations.
  - Tabular learning method (Q-table) is inefficient or infeasible when state-action space is very large or continuous, making it difficult to simulate the complexity of real market.
  - Sequential parameter updates make the learning samples temporally related, which likely to cause high bias in this experiment.

- For better market simulation, it's necessary to extend this study to more advanced and powerful RL algorithms.

## 🤖 Algorithm Framework: Deep Q-network
Deep Q-network (DQN) is a value-based RL algorithm that extends Q-learning by using deep neural networks (DNN) to approximate the optimal Q-function 
$Q^*(s,a)$, enabling it to handle complex, high-dimensional state-action space. It overcomes Q-learning's limitations through two key techniques:
  - `experience replay` - store and randomly sample past experiences, reducing temporal correlation and improving learning stability & data efficiency.
  - `target network` - fix and update parameters from main network periodically to avoid pursue moving goal, further enhancing the learning stability.

The pseudocode for DQN is provided below:
```
Initialize:
    - Main Q-network Q(s,a;θ) with random weights θ
    - Target Q-network Q^(s,a;θ^) with weights θ^ = θ
    - Replay memory D with capacity N

Set Hyperparameters:
    - Learning rate α
    - Discount factor γ
    - Target update frequency C
    - Exploration rate ε (ε-greedy policy) and its decay rate ε_decay

For episode = 1 to M do
    Initialize state s_1

    For t = 1 to T do:
                
        With probability ε select random action a_t
        
        Otherwise execute a_t = argmax_a Q(s_t,a;θ)
            
        Observe reward r_t and next state s_t+1

        Store transition (s_t, a_t, r_t, s_t+1) in D

        Sample random minibatch of transitions (s_j, a_j, r_j, s_j+1) from D
        
        Compute target Q-values:
            y_j = r_j, if episode terminates at step j+1
            y_j = r_j + γ * max_a' Q^(s_j+1, a'; θ^), otherwise

        Perform gradient descent on loss: L(θ) = 𝔼[(y_j - Q(s_j, a_j; θ))²] with respect to θ
        
        Update Q-network weights θ using optimizer
            
        Update target network periodically: If t mod C == 0: θ^ ← θ   
            
        Break if episode done
    
    Decay exploration rate: ε ← max(ε_min, ε · ε_decay)
    
End For
```
## 🧭 Experiment Design

### Economic Modelling: Logit Bertrand Competition
---
The market simulation is based on the **Logit Bertrand model** which describes oligopolistic price competition with considerations of product differentiation and heterogeneity of consumer preference to reflect complex market reality. In this model, consumers’ demand allocation across firms is based on a continuous probability distribution rather than an all-or-nothing setting that solely chooses the lowest-pricing product:

$$\Large\displaystyle d_i = \frac{e^{\frac{g - p_i}{\mu}}}{\sum_j e^{\frac{g - p_j}{\mu}} + e^0}$$

where:
- $d_i$ = firm i's demand (market share)
- $g$ = product quality
- $p_i$ = firm i's price
- $μ$ = product's substitutability
  - Lower μ → 0 means products are more substitutable, indicating fiercer price competition
  - Higher μ means products are more differentiated, indicating softer price competition
- $\frac{g - p_i}{\mu}$ normalizes the net utility of consumers from buying firm i’s product, with the parameter µ adjusting the price sensitivity of consumers. 

- $e^0$ = outside option (consumers can choose not to buy, meaning net utility = 0)

In this context, each firm employs an independent pricing algorithm, produces a product with a given quality $g$, incurs a marginal cost $c$, and sets product prices as $p_i$. Algorithms can set prices simultaneously, aiming to maximize their users' long-term profits. Consumers then choose products based on these prices. The profit for firm $i$ is given by,
$$ π_i = d_i × (p_i - c).$$

In the perfect competition setting, all firms will set price at the Nash-equilibrium level $p^N$, and have no incentive to unilaterally deviate from their pricing strategies. However, if firms coordinate their pricing strategies without any impediment, they would act like one single entity and maximize collective profits by pricing at monopoly level $p^M$ that is much higher than Nash price $p^N$.

In logit Bertrand model, firms determine the Nash price $p^N$ by optimizing their respective profit functions with respect to the price:
$$\frac{d}{dp_i}\pi_i=\frac{d}{dp_i}\left(\left(p_i-c_i\right)d_i\right)=0$$

To determine the monopoly price $p^M$, firms need to optimize the total profit:
$$\pi\left(p_0,p_1\right)=\sum_{i=0}^{1}{\left(p_i-c_i\right)d_i\left(p_i,p_{-i}\right)}$$ 

In practice, although analytical solutions to these functions are generally unavailable, numerical methods (eg. Newton–Raphson method) can be employed to approximate the solution.

\* Run independent ```benchmark_metrics.ipynb``` to calculate the price and profits at Nash-level and Monopoly-level with different parameters ($i, μ, g, c$).

### Evaluation Metrics
---
Two standardized metrics to quantify the degree of algorithmic collusion:
- ```RPDI (Relative Price Deviation Index)``` measures an agent’s pricing relative to Nash and monopoly levels, indicating the extent of supra-competitive pricing.
$$
RPDI = \frac{p̄ - p^N}{p^M - p^N}
$$

- ```Δ (Profit Metric)``` assesses an agent’s average profit over time, normalized relative to Nash and monopoly profits.

$$
Δ = \frac{π̄ - π^N}{π^M - π^N}
$$
- **RPDI, Δ → 0**: perfect competition (near Nash equilibrium)
- **RPDI, Δ ≈ 0.5**: incomplete competition
- **RPDI, Δ → 1**: collusive pricing (near monopoly level)


### Experiment Setting
---
#### **1. Basic DQN simulation**
- **Market Configuration**

    | Parameter | Value |
    |---|---|
    | Firm $i$ | 0, 1 |
    | Marginal cost $c$ | 1 |
    | Product quality $g$ | 2 |
    | Substitutability $µ$ | 0.25 |
    | Nash price $p^N$ | 1.473 |
    | Monopoly price $p^M$ | 1.925 |
    | Nash profit $π^N$ | 0.223 |
    | Monopoly profit $π^M$ | 0.337 |

- **Algorithmic Architecture**

	| Item | Spec |
	|---|---|
	| Agents | 2 DQN agents (representing 2 firms) |
	| Network | Fully connected with 2 hidden layers and 64 nodes each |
	| Activation | ReLU |
	| Optimizer | Adam |
	| Action space | Discrete prices $m=15$, evenly divided |
	| Price range* | [1.428, 1.970]|
	| Learning rate $\alpha$ | 0.01, 0.05, 0.1 |
    | Exploration rate $ε$ | start from 1.0 to 0.01 with ε_decay = 0.995 |

    \* Price range is $[\,p^N-ξ(p^M-p^N),\; p^M+ξ(p^M-p^N)\,]$, with $ξ$ set to 0.1 

- **Training Workflow**
    | Phase | Time | Description |
    |---|---|---|
    | Training | 2,000 episodes | Each episode represents a complete market trading cycle<br>Agents learn optimal pricing strategies through repeated decision-making<br>Agents explore and exploit using ε-greedy strategyε decays from 1.0 gradually to 0.01<br> 2,000 episodes to allow convergence of pricing behaviors|
    | Evaluation | 100 episodes | 100 post-training episodes are used to assess learning performance<br>During evaluation, agents use learned pricing strategies (with $ε$ set to minimum) |
    | Metric Calculation | last 10,000 timesteps | Only the last 10,000 timesteps of all 100 evaluation episodes are used to compute performance metrics RPDI and Δ to ensure stability and robustness |

- **Results Analysis**
    | Behavior | Metrics | Description | Market Characteristics |
    |---------------|------------|-------------|------------------------|
    | ✅ **Competitive** | RPDI, Δ < 0.3 | Prices near Nash equilibrium | • Agents compete fully<br>• Market operates efficiently<br>• No evidence of algorithmic collusion |
    | ⚠️ **Partial Coordination** | 0.3 ≤ RPDI, Δ ≤ 0.7 | Prices above Nash but below monopoly level | • Agents may coordinate pricing <br>• Moderate market power<br>• Antitrust concerns arise |
    | 🚨 **Collusive** | RPDI, Δ > 0.7 | Prices near monopoly level | • Indicate algorithmic collusion<br>• Warrants antitrust investigation|

#### **2. Extended Simulation**

After runnning the basic DQN simulation, the experiment will extend to 3 or 4 firms with symmetric or asymmetric model parameters, in the same algorithmic architecture and training workflow, to check the robustness in more complex market conditions.

- 3 firms with symmetric parameters
    | Parameter | Value |
    |---|---|
    | Firm $i$ | 0, 1, 2 |
    | Marginal cost $c$ | 1 |
    | Product quality $g$ | 2 |
    | Substitutability $µ$ | 0.25 |
    | Nash price $p^N$ | 1.370 |
    | Monopoly price $p^M$ | 2.000 |
    | Nash profit $π^N$ | 0.120 |
    | Monopoly profit $π^M$ | 0.250 |
    | Price range | [1.307, 2.063] |


- 4 firms with symmetric parameters
    | Parameter | Value |
    |---|---|
    | Firm $i$ | 0, 1, 2, 3 |
    | Marginal cost $c$ | 1 |
    | Product quality $g$ | 2 |
    | Substitutability $µ$ | 0.25 |
    | Nash price $p^N$ | 1.331 |
    | Monopoly price $p^M$ | 2.054 |
    | Nash profit $π^N$ | 0.081 |
    | Monopoly profit $π^M$ | 0.201 |
    | Price range | [1.259, 2.126] |

- 4 firms with asymmetric parameters
    | Parameter | Value |
    |---|---|
    | Firm $i$ | 0, 1, 2, 3 |
    | Marginal cost $c$ | [1.05, 1.10, 0.95, 1.00] |
    | Product quality $g$ | [2.10, 2.00, 1.90, 1.80] |
    | Substitutability $µ$ | 0.30 |
    | Nash price $p^N$ | [1.486, 1.486, 1.351, 1.363] |
    | Monopoly price $p^M$ | [2.121, 2.171, 2.021, 2.071] |
    | Nash profit $π^N$ | [0.136, 0.086, 0.101, 0.063] |
    | Monopoly profit $π^M$ | [0.280, 0.170, 0.200, 0.122] |
    | Price range | [1.423, 2.185], [1.418, 2.240], [1.284, 2.088], [1.292, 2.142] |

## 🚀 Quick Start Guide

### Environment Configuration
---

```bash
# Clone the repository
git clone <repository-url>
cd DQN-simulation

# Install dependencies
pip install -r requirements.txt
```

### Running Experiments
---

**Option 1: Interactive Jupyter Notebook (Recommended)**

```bash
jupyter notebook main_experiment.ipynb
```

Run cells sequentially to:
- Explore each experiment interactively
- Visualize training workflow in real-time
- Analyze experiment results and their economic significance

**Best for**: learning, experimentation, and detailed analysis

**Option 2: Run All Experiments at Once**

```bash
python run_experiments.py
```

This operation will:
- Run all 4 experiments automatically
- Generate comprehensive results and visualizations
- Save results in the `results/` directory

**Best for**: skip training details and conduct overall analysis

**Option 3: Quick Test Mode**

For rapid testing with fewer episodes:

```bash
python run_experiments.py --quick
```

This runs all experiments with only 500 episodes (vs 2000 default).

**Best for**: developer testing and results preview

### Calculate Benchmark Metrics
---

To verify Nash-equilibrium and Monopoly-level calculations:

```bash
jupyter notebook benchmark_metrics.ipynb
```

Run all cells to see numerical solutions for each scenario.

### Results
---

After running experiments, find results in:

- `results/` - Main results directory
  - `exp1_2firm/` - 2-firm symmetric results
  - `exp2_3firm/` - 3-firm symmetric results
  - `exp3_4firm/` - 4-firm symmetric results
  - `exp4_4firm_asym/` - 4-firm asymmetric results
  - `summary_*.csv` - Comparative summary
  - `comparative_analysis.png` - Visualization

Each experiment folder contains:
- `training_history.png` - Overall training metrics (6 subplots)
- `price_profit_dynamics.png` - **Detailed line charts showing price and profit evolution across all training episodes for each firm, with Nash and Monopoly benchmarks as reference lines**
- `evaluation_results.json` - Final metrics (RPDI, Δ)
- `checkpoint_*/` - Model checkpoints

## 📁 Project Structure

```
DQN-simulation/
├── experiment_config.py      # All experiment configurations
├── logit_market_env.py       # Logit Bertrand market environment
├── dqn_agents.py             # DQN agent configuration
├── market_simulation.py      # Complete simulation process
├── run_experiments.py        # Running all experiments at once
├── main_experiment.ipynb     # Interactive notebook for step-by-step experiment
├── benchmark_metrics.ipynb   # Nash/Monopoly calculations
├── requirements.txt          # Python environment dependencies
└── results/                  # Output directory
```


## 📚 Reference
- Calvano, E., Calzolari, G., Denicolò, V., & Pastorello, S. (2020). *Artificial Intelligence, Algorithmic Pricing, and Collusion.* American Economic Review 110 (10): 3267–97.
- Deng, S., Schiffer, M., & Bichler, M. (2025). *Exploring Competitive and Collusive Behaviors in Algorithmic Pricing with Deep Reinforcement Learning.* arXiv:2503.11270
- Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). *Playing Atari with Deep Reinforcement Learning.* https://arxiv.org/abs/1312.5602

---
**Author:** Zhou Ziyue (William)<br>

**Last Updated:** 2025-10-22<br>

**License:** Copyright © [2025] [Zhou Ziyue]
All Rights Reserved.

Permission is granted to view and cite this work for academic purposes only.
Modification and redistribution are not permitted without explicit permission.

