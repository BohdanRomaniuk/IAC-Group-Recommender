"""
Plotting utilities for training / evaluation loss curves.
"""
import os
from datetime import datetime
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless environments
import matplotlib.pyplot as plt


def plot_losses(
    train_actor_losses: List[Optional[float]],
    train_critic_losses: List[Optional[float]],
    eval_actor_losses: List[float],
    eval_critic_losses: List[float],
    eval_loss_steps: List[int],
    save_dir: str = "results",
    filename: Optional[str] = None,
) -> str:
    """
    Save a two-panel PNG showing actor and critic loss curves.
    """
    os.makedirs(save_dir, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}_loss_curves.png"

    save_path = os.path.join(save_dir, filename)

    train_actor_x = [i for i, v in enumerate(train_actor_losses) if v is not None]
    train_actor_y = [v for v in train_actor_losses if v is not None]
    train_critic_x = [i for i, v in enumerate(train_critic_losses) if v is not None]
    train_critic_y = [v for v in train_critic_losses if v is not None]

    fig, (ax_actor, ax_critic) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("DDPG Training — Actor & Critic Loss Curves", fontsize=14, fontweight="bold")

    # ── Actor panel ──────────────────────────────────────────────────────────
    ax_actor.set_title("Actor Loss")
    ax_actor.set_xlabel("Episode")
    ax_actor.set_ylabel("Loss")

    if train_actor_y:
        ax_actor.plot(
            train_actor_x, train_actor_y,
            color="green", linestyle="-", linewidth=1.2, alpha=0.85,
            label="Train",
        )
    if eval_actor_losses and eval_loss_steps:
        ax_actor.plot(
            eval_loss_steps, eval_actor_losses,
            color="red", linestyle="--", linewidth=2.0,
            marker="o", markersize=5,
            label="Evaluation",
        )
    ax_actor.legend(loc="upper right")
    ax_actor.grid(True, linestyle=":", alpha=0.5)

    # ── Critic panel ─────────────────────────────────────────────────────────
    ax_critic.set_title("Critic Loss")
    ax_critic.set_xlabel("Episode")
    ax_critic.set_ylabel("Loss")

    if train_critic_y:
        ax_critic.plot(
            train_critic_x, train_critic_y,
            color="green", linestyle="-", linewidth=1.2, alpha=0.85,
            label="Train",
        )
    if eval_critic_losses and eval_loss_steps:
        ax_critic.plot(
            eval_loss_steps, eval_critic_losses,
            color="red", linestyle="--", linewidth=2.0,
            marker="o", markersize=5,
            label="Evaluation",
        )
    ax_critic.legend(loc="upper right")
    ax_critic.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Loss curves saved → {os.path.abspath(save_path)}")
    return os.path.abspath(save_path)

