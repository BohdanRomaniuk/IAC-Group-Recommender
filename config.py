"""
Configurations
"""
import os
from typing import Optional

import torch

try:
    import yaml  # type: ignore
except ImportError:  # PyYAML is optional; from_yaml will raise if missing
    yaml = None  # type: ignore


class Config(object):
    """
    Configurations
    """

    def __init__(self):
        # Data
        self.data_folder_path = os.path.join('./', 'data', 'MovieLens-Rand')  # Path to the dataset folder
        self.item_path = os.path.join(self.data_folder_path, 'movies.dat')    # Path to the movies file
        self.user_path = os.path.join(self.data_folder_path, 'users.dat')     # Path to the users file
        self.group_path = os.path.join(self.data_folder_path, 'groupMember.dat')  # Path to the group-members mapping file
        self.saves_folder_path = os.path.join('data', 'saves')                # Folder where trained model checkpoints are saved
        self.results_folder_path = os.path.join('data', 'results')            # Folder for loss-curve plots and metric logs

        # Recommendation system
        self.history_length = 10                     # Number of recently watched items included in the state (sliding window)
        self.top_K_list = [5, 10, 15, 20]   # Values of K used for Precision@K and Recall@K evaluation metrics
        self.rewards = [0, 1]                        # Possible reward values: 0 = item not liked, 1 = item liked

        # Reinforcement learning
        self.embedding_size = 64                                               # Dimensionality of user and item embedding vectors
        self.state_size = self.history_length + 1                              # State size: history_length items + 1 aggregated group vector
        self.action_size = 1                                                   # Number of items recommended per step
        self.embedded_state_size = self.state_size * self.embedding_size       # Total state vector size fed into Actor/Critic (11 * 64 = 704)
        self.embedded_action_size = self.action_size * self.embedding_size     # Total action vector size (1 * 64 = 64)

        # Numbers
        self.item_num = None         # Total number of unique items (movies); filled at runtime from dataset
        self.user_num = None         # Total number of unique users; filled at runtime from dataset
        self.group_num = None        # Number of groups used for training; filled at runtime from dataset
        self.total_group_num = None  # Total number of groups (train + eval); filled at runtime from dataset

        # Environment
        self.env_n_components = self.embedding_size  # Number of latent components in NMF (matches embedding_size)
        self.env_tol = 1e-4                          # Convergence tolerance for NMF: stops if update < 0.0001
        self.env_max_iter = 1000                     # Maximum number of NMF iterations to prevent infinite loops
        self.env_alpha = 0.001                       # Regularization coefficient for NMF to prevent overfitting

        # Actor-Critic network
        self.actor_hidden_sizes = (256, 128)  # Hidden layer sizes of Actor MLP:  704 → 256 → 128 → 64
        self.critic_hidden_sizes = (256, 128) # Hidden layer sizes of Critic MLP: 768 → 256 → 128 → 1

        # DDPG algorithm
        self.tau = 5e-3   # Soft update coefficient for target networks: slightly larger for faster target sync
        self.gamma = 0.95 # Discount factor for future rewards: slightly higher to value long-term more

        # Optimizer
        self.batch_size = 128                   # Number of transitions sampled from replay buffer per training update
        self.buffer_size = 100000               # Maximum number of (s, a, r, s') transitions stored in replay memory
        self.num_episodes = 1000                # Total number of training episodes
        self.num_steps = 100                    # Number of recommendation steps per episode (total steps = 1000 * 100 = 100 000)
        self.num_updates_per_step = 2           # Number of gradient updates per environment step
        self.warmup_steps = 500                 # Fill buffer with random actions before any gradient update
        self.embedding_weight_decay = 1e-6      # L2 regularization for Embedding optimizer (prevents large weights)
        self.actor_weight_decay = 1e-6          # L2 regularization for Actor optimizer
        self.critic_weight_decay = 1e-6         # L2 regularization for Critic optimizer
        self.embedding_learning_rate = 5e-4     # Learning rate for Adam optimizer of Embedding network
        self.actor_learning_rate = 1e-3         # Learning rate for Adam optimizer of Actor network
        self.critic_learning_rate = 1e-3        # Learning rate for Adam optimizer of Critic network
        self.eval_per_iter = 10                 # Run evaluation every N episodes

        # OU noise (Ornstein-Uhlenbeck process — correlated noise for continuous action space exploration)
        self.ou_mu = 0.0      # Mean value the noise reverts to over time
        self.ou_theta = 0.15  # Speed of mean reversion: how quickly noise returns to mu
        self.ou_sigma = 0.3   # Volatility: amplitude of random fluctuations (increased for more exploration)
        self.ou_epsilon = 1.0 # Initial noise scale (can be decayed over time to reduce exploration)

        # Entropy regularisation for Actor (prevents overconfident / collapsing policy)
        self.entropy_coef = 0.05  # Weight of the entropy bonus added to the actor loss (increased)

        # Reward shaping coefficients
        self.reward_diversity_alpha = 0.5   # Weight of diversity bonus (unique items in recommendation window)
        self.reward_coverage_beta = 0.1     # Weight of coverage bonus (novel items not in history)

        # Prioritized Experience Replay
        self.per_alpha = 0.6            # Priority exponent (0 = uniform, 1 = full prioritisation)
        self.per_beta_start = 0.4       # Initial IS-weight exponent (annealed to 1 over training)
        self.per_beta_frames = 100_000  # Steps over which beta is annealed to 1.0

        # GPU
        if torch.cuda.is_available():
            print("Using GPU (CUDA acceleration enabled)")
            self.device = torch.device("cuda:0")
        else:
            print("Using CPU")
            self.device = torch.device("cpu")

    # ------------------------------------------------------------------ #
    @classmethod
    def from_yaml(cls, path: Optional[str]) -> "Config":
        """
        Build a Config and override fields from a YAML file.

        Any keys missing from the YAML keep their default values defined
        in __init__. Tuple-valued fields (e.g. ``actor_hidden_sizes``) are
        cast back to tuples for consistency. ``device`` is always
        re-resolved at runtime regardless of YAML.

        :param path: path to YAML file, or None to return defaults.
        :return: Config instance
        """
        config = cls()
        if not path:
            return config
        if yaml is None:
            raise ImportError("PyYAML is required for Config.from_yaml(). "
                              "Install with: pip install pyyaml")
        with open(path, 'r') as fh:
            data = yaml.safe_load(fh) or {}

        tuple_fields = {"actor_hidden_sizes", "critic_hidden_sizes"}
        for key, value in data.items():
            if not hasattr(config, key):
                continue
            if key in tuple_fields and isinstance(value, list):
                value = tuple(value)
            setattr(config, key, value)

        # Recompute derived values that depend on overridden fields
        config.state_size = config.history_length + 1
        config.embedded_state_size = config.state_size * config.embedding_size
        config.embedded_action_size = config.action_size * config.embedding_size
        return config

