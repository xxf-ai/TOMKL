"""Shared and behavior-specific mixture of experts for BASE.

The three tasks correspond to purchase, favorite/cart, and view. Each task
mixes the same shared experts with its own behavior-specific experts.
"""

import torch
from torch import nn
from torch.nn import functional as F


class mmoe_layer(nn.Module):
    """PyTorch equivalent of the original BASE expert/gate layer."""

    def __init__(
        self,
        input_dim,
        hidden_units,
        num_expert_sh,
        num_expert_sp,
        num_tasks=3,
        dropout_rate=0.0,
        use_expert_bias=True,
        use_gate_bias=True,
    ):
        super().__init__()
        if num_tasks != 3:
            raise ValueError("BASE expects exactly three behavior tasks")
        if num_expert_sh + num_expert_sp < 1:
            raise ValueError("At least one expert is required")

        self.hidden_units = hidden_units
        self.num_expert_sh = num_expert_sh
        self.num_expert_sp = num_expert_sp
        self.num_tasks = num_tasks
        self.dropout = nn.Dropout(dropout_rate)

        def expert_weight(count):
            weight = nn.Parameter(torch.empty(input_dim, hidden_units, count))
            nn.init.trunc_normal_(weight, std=(2.0 / (3 * hidden_units + 1)) ** 0.5)
            return weight

        self.shared_weight = expert_weight(num_expert_sh) if num_expert_sh else None
        self.specific_weights = nn.ParameterList(
            [expert_weight(num_expert_sp) for _ in range(num_tasks)]
        ) if num_expert_sp else nn.ParameterList()
        self.shared_bias = (
            nn.Parameter(torch.zeros(hidden_units, num_expert_sh))
            if use_expert_bias and num_expert_sh else None
        )
        self.specific_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(hidden_units, num_expert_sp))
             for _ in range(num_tasks)]
        ) if use_expert_bias and num_expert_sp else nn.ParameterList()
        self.gates = nn.ModuleList(
            [nn.Linear(input_dim, num_expert_sh + num_expert_sp, bias=use_gate_bias)
             for _ in range(num_tasks)]
        )
        for gate in self.gates:
            nn.init.trunc_normal_(gate.weight, std=(2.0 / (3 * hidden_units + 1)) ** 0.5)
            if gate.bias is not None:
                nn.init.zeros_(gate.bias)

    def forward(self, inputs, target_inputs=None):
        if inputs.ndim != 3:
            raise ValueError(f"Expert input must be [batch, length, dim], got {tuple(inputs.shape)}")
        shared = None
        if self.shared_weight is not None:
            shared = torch.einsum("bli,ihn->blhn", inputs, self.shared_weight)
            if self.shared_bias is not None:
                shared = shared + self.shared_bias
            shared = F.relu(shared)

        outputs = []
        for task, gate in enumerate(self.gates):
            expert_parts = []
            if shared is not None:
                expert_parts.append(shared)
            if self.num_expert_sp:
                specific = torch.einsum("bli,ihn->blhn", inputs, self.specific_weights[task])
                if len(self.specific_biases):
                    specific = specific + self.specific_biases[task]
                expert_parts.append(F.relu(specific))
            experts = torch.cat(expert_parts, dim=-1)
            weights = self.dropout(F.softmax(gate(inputs), dim=-1))
            outputs.append((experts * weights.unsqueeze(-2)).sum(dim=-1))
        return outputs


class tower_layer(nn.Module):
    """One behavior tower: ReLU hidden layer followed by linear output."""

    def __init__(self, input_dim, hidden_units, output_dim, activation="relu"):
        super().__init__()
        if activation != "relu":
            raise ValueError(f"Unsupported tower activation: {activation}")
        self.hidden_layer = nn.Linear(input_dim, hidden_units)
        self.output_layer = nn.Linear(hidden_units, output_dim)

    def forward(self, inputs, target_behavior_emb=None):
        return self.output_layer(F.relu(self.hidden_layer(inputs)))
