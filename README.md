# IAC-Group-Recommender

Integrator-Actor-Critic RL Group Recommendation System.

A DDPG-based reinforcement-learning recommender for **groups** of users, built
on MovieLens. Uses an **attention-based group integrator**, a **prioritized
replay buffer**, **entropy-regularised** actor updates, and **cosine-annealed**
Ranger optimisers.

---

## Project Structure

```
IAC-Group-Recommender/
├── main.py                 # Thin entrypoint — parses --config, builds components, runs Trainer
├── config.py               # Config dataclass + Config.from_yaml()
├── default.yaml            # Default hyperparameters (override via --config)
│
├── models/                 # Neural networks (one class per file)
│   ├── actor.py            # Actor MLP
│   ├── critic.py           # Critic MLP
│   └── integrator.py       # User/item Embedding + Attention group aggregator
│
├── agent/                  # RL agent
│   ├── ddpg.py             # DDPGAgent (training step, action selection, eval loss)
│   └── replay_buffer.py    # PrioritizedReplayBuffer
│
├── environment/            # Environment + data + training/eval orchestration
│   ├── env.py              # gymnasium.Env wrapping a NMF-predicted rating matrix
│   ├── dataloader.py       # Reads MovieLens groups / ratings / negative samples
│   ├── trainer.py          # Trainer class — full training loop with checkpointing
│   └── evaluator.py        # Recall@K, NDCG@K, Precision@K, MAP, HR, MRR, Coverage
│
├── utils/
│   ├── noise.py            # Ornstein-Uhlenbeck exploration noise
│   └── helpers.py          # Simple uniform ReplayMemory (kept for reference)
│
├── visualization/
│   └── plot.py             # plot_losses() — actor/critic loss curves
│
├── generators/             # Dataset preprocessing scripts
│   ├── generator.py        # MovieLens-1M
│   └── generator_32m.py    # MovieLens-32M
│
├── data/                   # All data, checkpoints, and results
│   ├── ml-1m/  ml-32m/  MovieLens-Rand/  MovieLens-32m/
│   ├── saves/              # Best model checkpoints + cached eval pickles + NMF env
│   └── results/            # Loss curves and metric logs
│
├── requirements.txt
└── README.md
```

> **Note:** This project uses **implicit namespace packages** — there are no
> `__init__.py` files. Python 3 supports them natively; run all commands from
> the project root.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### Run with built-in defaults
```bash
python main.py
```

### Run with a custom YAML config
```bash
python main.py --config default.yaml
```

Any field in the YAML overrides the matching default in `config.py`; missing
fields keep their defaults. Tuple-valued fields (e.g. `actor_hidden_sizes:
[256, 128]`) are automatically cast back to tuples.

### Generate group datasets (one-time)
```bash
python -m generators.generator      # MovieLens-1M
python -m generators.generator_32m  # MovieLens-32M
```

---

## Key Components

| Component                  | File                          | Purpose                              |
| -------------------------- | ----------------------------- | ------------------------------------ |
| `Actor`                    | `models/actor.py`             | Maps state → action embedding        |
| `Critic`                   | `models/critic.py`            | Estimates Q(s, a)                    |
| `Embedding` (Integrator)   | `models/integrator.py`        | User/item embeddings + attention pool|
| `DDPGAgent`                | `agent/ddpg.py`               | Training, action selection, eval     |
| `PrioritizedReplayBuffer`  | `agent/replay_buffer.py`      | TD-error-weighted sampling           |
| `Env`                      | `environment/env.py`          | gymnasium env over NMF rating matrix |
| `DataLoader`               | `environment/dataloader.py`   | Reads MovieLens datasets             |
| `Trainer`                  | `environment/trainer.py`      | End-to-end training loop             |
| `Evaluator`                | `environment/evaluator.py`    | Top-K ranking metrics                |
| `OUNoise`                  | `utils/noise.py`              | Exploration noise                    |
| `plot_losses`              | `visualization/plot.py`       | PNG of actor/critic loss curves      |

---

## Outputs

After training:
- `data/saves/best_actor.pt`, `data/saves/best_critic.pt`, `data/saves/best_embedding.pt` —
  best checkpoint by **Group Recall@10**
- `data/saves/env_<split>_<dim>.npy` — cached NMF rating-matrix prediction
- `data/saves/eval_{user,group}_{val,test}_<H>.pkl` — cached evaluation dataframes
- `data/results/<timestamp>_loss_curves.png` — actor + critic loss curves
