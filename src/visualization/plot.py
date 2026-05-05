"""Loss-curve plotting for DDPG training runs.

A single helper, :func:`plot_losses`, writes a two-panel PNG (actor / critic)
under ``save_dir``. The non-interactive ``Agg`` backend is used so this is
safe to call from headless environments (CI, remote servers, …).
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)


__all__ = ["plot_losses"]

logger = logging.getLogger(__name__)


# Style constants
_TRAIN_STYLE = dict(color="green", linestyle="-", linewidth=1.2, alpha=0.85, label="Train")
_EVAL_STYLE = dict(color="red", linestyle="--", linewidth=2.0, marker="o", markersize=5, label="Evaluation")


def _filter_non_none(
    series: Sequence[Optional[float]],
) -> tuple[list[int], list[float]]:
    """Drop ``None`` entries while keeping the original indices."""
    xs = [i for i, v in enumerate(series) if v is not None]
    ys = [v for v in series if v is not None]
    return xs, ys


def _draw_panel(
    ax: plt.Axes,
    title: str,
    train_series: Sequence[Optional[float]],
    eval_series: Sequence[float],
    eval_steps: Sequence[int],
) -> None:
    """Render one (actor or critic) loss panel."""
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")

    xs, ys = _filter_non_none(train_series)
    if ys:
        ax.plot(xs, ys, **_TRAIN_STYLE)
    if eval_series and eval_steps:
        ax.plot(eval_steps, eval_series, **_EVAL_STYLE)

    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.5)


def plot_losses(
    train_actor_losses: List[Optional[float]],
    train_critic_losses: List[Optional[float]],
    eval_actor_losses: List[float],
    eval_critic_losses: List[float],
    eval_loss_steps: List[int],
    save_dir: str = "results",
    filename: Optional[str] = None,
) -> str:
    """Save a two-panel PNG of actor & critic loss curves.

    Args:
        train_actor_losses: Per-episode mean actor loss (``None`` allowed).
        train_critic_losses: Per-episode mean critic loss (``None`` allowed).
        eval_actor_losses: Sparse evaluation-time actor losses.
        eval_critic_losses: Sparse evaluation-time critic losses.
        eval_loss_steps: Episode indices for the evaluation losses.
        save_dir: Output directory (created if missing).
        filename: Custom filename; a timestamped one is used by default.

    Returns:
        Absolute path of the saved PNG.
    """
    target_dir = Path(save_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}_loss_curves.png"

    save_path = (target_dir / filename).resolve()

    fig, (ax_actor, ax_critic) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("DDPG Training — Actor & Critic Loss Curves", fontsize=14, fontweight="bold")

    _draw_panel(ax_actor, "Actor Loss", train_actor_losses, eval_actor_losses, eval_loss_steps)
    _draw_panel(ax_critic, "Critic Loss", train_critic_losses, eval_critic_losses, eval_loss_steps)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Loss curves saved → %s", save_path)
    return str(save_path)

