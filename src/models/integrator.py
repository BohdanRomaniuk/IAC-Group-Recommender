"""State integrator network with attention-based group aggregation.

The :class:`Integrator` wraps two ``nn.Embedding`` tables (users + items)
and a :class:`MultiHeadGroupAggregation` module that fuses individual member
preferences into a single group representation.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


__all__ = [
    "Integrator",
    "MultiHeadGroupAggregation",
]


class MultiHeadGroupAggregation(nn.Module):
    """Aggregate member embeddings via multi-head self-attention + gated residual.

    Replaces a naive weighted-mean by letting members attend to one another
    before being averaged, while a learned gate controls how strongly the
    attended representation overrides the original embeddings.
    """

    def __init__(
        self,
        user_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """Initialise the aggregator.

        Args:
            user_dim: Dimensionality of each member embedding.
            num_heads: Number of attention heads. Falls back to 1 when
                ``user_dim`` is not divisible by ``num_heads``.
            dropout: Dropout used inside the multi-head attention block.
        """
        super().__init__()

        if user_dim % num_heads != 0:
            num_heads = 1

        self.attention = nn.MultiheadAttention(
            embed_dim=user_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(user_dim * 2, user_dim),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------ #
    def forward(
        self,
        user_embeddings: torch.Tensor,
        group_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Aggregate ``[group_size, user_dim]`` member embeddings.

        Args:
            user_embeddings: Tensor of shape ``[group_size, user_dim]``.
            group_mask: Optional boolean padding mask passed through to the
                attention layer.

        Returns:
            A single ``[user_dim]`` vector representing the whole group.
        """
        x = user_embeddings.unsqueeze(0)
        attended, _ = self.attention(x, x, x, key_padding_mask=group_mask)
        attended = attended.squeeze(0)

        gate = self.gate(torch.cat([attended, user_embeddings], dim=-1))
        fused = gate * attended + (1.0 - gate) * user_embeddings

        return fused.mean(dim=0)


class Integrator(nn.Module):
    """User/item embedding tables + attention-based group integration.

    The forward pass returns the *embedded state*, formed by concatenating
    the aggregated group vector with the flattened item-history embedding.
    """

    def __init__(
        self,
        embedding_size: int,
        user_num: int,
        item_num: int,
        num_heads: int = 4,
    ) -> None:
        """Create the embedding tables and group aggregator.

        Args:
            embedding_size: Dimensionality of every user/item vector.
            user_num: Total number of users (largest user id).
            item_num: Total number of items (largest item id).
            num_heads: Attention heads used for group aggregation.
        """
        super().__init__()

        self.user_embedding = nn.Embedding(user_num + 1, embedding_size)
        self.item_embedding = nn.Embedding(item_num + 1, embedding_size)
        self._init_embeddings()

        self.group_aggregator = MultiHeadGroupAggregation(
            user_dim=embedding_size,
            num_heads=num_heads,
        )

    # ------------------------------------------------------------------ #
    def _init_embeddings(self) -> None:
        """Apply small-std Gaussian init to embedding tables."""
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        group_members: torch.Tensor,
        history: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a single (group, history) pair into a state vector.

        Args:
            group_members: 1-D tensor of user ids belonging to the group.
            history: 1-D tensor of recent item ids (length =
                ``history_length``).

        Returns:
            The embedded state vector.
        """
        member_vectors = self.user_embedding(group_members)
        group_vector = self.group_aggregator(member_vectors)
        history_vector = torch.flatten(self.item_embedding(history), start_dim=-2)
        return torch.cat([group_vector, history_vector], dim=-1)




