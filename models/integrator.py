"""
Embedding / Integrator network with attention-based group aggregation.
"""
import torch
import torch.nn as nn


class AttentionGroupAggregator(nn.Module):
    """
    Replaces simple weighted-mean aggregation with Multi-Head Self-Attention
    followed by a gated residual connection so that individual member
    preferences are preserved while the model learns interaction patterns.
    """

    def __init__(self, user_dim: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Initialize AttentionGroupAggregator

        :param user_dim: dimensionality of each user embedding
        :param num_heads: number of attention heads
        :param dropout: dropout probability inside attention
        """
        super(AttentionGroupAggregator, self).__init__()

        # num_heads must divide user_dim evenly; fall back to 1 if not
        if user_dim % num_heads != 0:
            num_heads = 1

        self.attention = nn.MultiheadAttention(
            embed_dim=user_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Gating: decide how much of the attended representation to use
        self.gate = nn.Sequential(
            nn.Linear(user_dim * 2, user_dim),
            nn.Sigmoid(),
        )

    def forward(self, user_embeddings: torch.Tensor, group_mask=None) -> torch.Tensor:
        """
        Forward

        :param user_embeddings: [group_size, user_dim]
        :param group_mask: optional boolean padding mask
        :return: group_repr [user_dim]
        """
        x = user_embeddings.unsqueeze(0)
        attn_out, _ = self.attention(x, x, x, key_padding_mask=group_mask)
        attn_out = attn_out.squeeze(0)

        gate = self.gate(torch.cat([attn_out, user_embeddings], dim=-1))
        fused = gate * attn_out + (1 - gate) * user_embeddings  # gated residual

        group_repr = fused.mean(dim=0)
        return group_repr


class Embedding(nn.Module):
    """
    Integrator/Embedding network with attention-based group aggregation.
    """

    def __init__(self, embedding_size: int, user_num: int, item_num: int, num_heads: int = 4):
        """
        Initialize Embedding

        :param embedding_size: embedding size
        :param user_num: number of users
        :param item_num: number of items
        :param num_heads: number of attention heads for group aggregation
        """
        super(Embedding, self).__init__()
        self.user_embedding = nn.Embedding(user_num + 1, embedding_size)
        self.item_embedding = nn.Embedding(item_num + 1, embedding_size)

        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)

        self.group_aggregator = AttentionGroupAggregator(
            user_dim=embedding_size,
            num_heads=num_heads,
        )

    def forward(self, group_members: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """
        Forward

        :param group_members: 1-D tensor of user ids
        :param history: 1-D tensor of item ids
        :return: embedded state
        """
        embedded_group_members = self.user_embedding(group_members)
        embedded_group = self.group_aggregator(embedded_group_members)
        embedded_history = torch.flatten(self.item_embedding(history), start_dim=-2)
        embedded_state = torch.cat([embedded_group, embedded_history], dim=-1)
        return embedded_state

