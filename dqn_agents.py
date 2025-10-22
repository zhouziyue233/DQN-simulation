"""
DQN Agents for Market Price Competition

This module implements Deep Q-Network agents following the experimental specifications
from the README, including experience replay and target networks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from typing import Dict, Optional, Tuple

# Check device availability
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using MPS (Metal Performance Shaders) for GPU acceleration")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA for GPU acceleration")
else:
    device = torch.device('cpu')
    print("Using CPU")


class DQNNetwork(nn.Module):
    """
    Deep Q-Network with 2 hidden layers of 64 nodes each
    Architecture follows experimental specification
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        """
        Initialize DQN network
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            hidden_size: Number of nodes in hidden layers (default: 64)
        """
        super(DQNNetwork, self).__init__()
        
        # Two hidden layers with ReLU activation
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x: Input state tensor
            
        Returns:
            Q-values for each action
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions
    """
    
    def __init__(self, buffer_size: int, batch_size: int):
        """
        Initialize replay buffer
        
        Args:
            buffer_size: Maximum number of transitions to store
            batch_size: Number of transitions to sample per batch
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", 
                                    field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to the buffer"""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self) -> Tuple:
        """
        Sample a batch of experiences
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) tensors
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])
        ).float().to(device)
        
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])
        ).long().to(device)
        
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])
        ).float().to(device)
        
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])
        ).float().to(device)
        
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        ).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """Return the current size of the buffer"""
        return len(self.memory)


class DQNAgent:
    """
    DQN Agent for pricing decisions in market competition
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        config: Dict,
        agent_id: int = 0,
        seed: int = 42,
    ):
        """
        Initialize DQN agent
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            config: DQN hyperparameters configuration
            agent_id: Unique identifier for the agent
            seed: Random seed for reproducibility
        """
        self.state_size = state_size
        self.action_size = action_size
        self.agent_id = agent_id
        self.seed = random.seed(seed)
        
        # Extract hyperparameters from config
        self.learning_rate = config['learning_rate']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.epsilon = config['epsilon_start']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_decay = config['epsilon_decay']
        self.buffer_size = config['buffer_size']
        self.batch_size = config['batch_size']
        self.hidden_size = config['hidden_size']
        self.update_frequency = config['update_frequency']
        
        # Initialize Q-networks
        self.q_network = DQNNetwork(state_size, action_size, self.hidden_size).to(device)
        self.target_network = DQNNetwork(state_size, action_size, self.hidden_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        
        # Initialize target network with same weights as main network
        self.hard_update_target_network()
        
        # Training statistics
        self.steps_done = 0
        self.losses = []
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select an action using ε-greedy policy
        
        Args:
            state: Current state observation
            training: Whether in training mode (uses exploration)
            
        Returns:
            Selected action index
        """
        # Epsilon-greedy action selection
        if training and random.random() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        # Exploit: choose action with highest Q-value
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return np.argmax(q_values.cpu().data.numpy())
    
    def step(self, state, action, reward, next_state, done):
        """
        Store experience and learn if enough samples available
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        # Store experience in replay buffer
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every update_frequency steps
        self.steps_done += 1
        if self.steps_done % self.update_frequency == 0:
            if len(self.memory) >= self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)
    
    def learn(self, experiences: Tuple):
        """
        Update Q-network using batch of experiences
        
        Args:
            experiences: Tuple of (states, actions, rewards, next_states, dones)
        """
        states, actions, rewards, next_states, dones = experiences
        
        # Get current Q values for chosen actions
        q_current = self.q_network(states).gather(1, actions)
        
        # Get max Q values for next states from target network
        q_next_max = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute target Q values
        q_targets = rewards + (self.gamma * q_next_max * (1 - dones))
        
        # Compute loss (MSE between current and target Q values)
        loss = F.mse_loss(q_current, q_targets)
        
        # Optimize the network
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Store loss for tracking
        self.losses.append(loss.item())
        
        # Soft update target network
        self.soft_update_target_network()
    
    def soft_update_target_network(self):
        """
        Soft update of target network parameters:
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def hard_update_target_network(self):
        """
        Hard update: copy all weights from main network to target network
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """
        Decay exploration rate
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str):
        """
        Save agent's model and optimizer state
        
        Args:
            filepath: Path to save the model
        """
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'agent_id': self.agent_id,
        }
        torch.save(checkpoint, filepath)
    
    def load(self, filepath: str):
        """
        Load agent's model and optimizer state
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.steps_done = checkpoint.get('steps_done', 0)
    
    def get_statistics(self) -> Dict:
        """
        Get training statistics
        
        Returns:
            Dictionary with training statistics
        """
        return {
            'agent_id': self.agent_id,
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'buffer_size': len(self.memory),
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
        }