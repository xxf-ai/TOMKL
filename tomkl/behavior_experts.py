"""Behavior-specific mixture of experts and towers used by BASE."""

from torch import nn

from .expert_layers import mmoe_layer, tower_layer


class TBRE(nn.Module):
    """Three purchase/favorite/view towers over shared and specific experts."""

    def __init__(
        self,
        maxlen,
        mmoe_hidden_units,
        num_experts_sh,
        num_experts_sp,
        num_tasks,
        tower_hidden_units,
        output_dim,
        behavior_id=None,
        dropout_rate=0.0,
        l2_reg=0.0,
        is_training=None,
        activation="relu",
        use_expert_bias=True,
        use_gate_bias=True,
    ):
        super().__init__()
        self.maxlen = maxlen
        self.num_tasks = num_tasks
        self.num_units = output_dim
        self.mmoe_layer = mmoe_layer(
            input_dim=3 * output_dim,
            hidden_units=mmoe_hidden_units,
            num_expert_sh=num_experts_sh,
            num_expert_sp=num_experts_sp,
            num_tasks=num_tasks,
            dropout_rate=dropout_rate,
            use_expert_bias=use_expert_bias,
            use_gate_bias=use_gate_bias,
        )
        self.tower_layer = nn.ModuleList(
            [tower_layer(mmoe_hidden_units, tower_hidden_units, output_dim, activation)
             for _ in range(num_tasks)]
        )
        self.normalizations = nn.ModuleList(
            [nn.LayerNorm(output_dim, eps=1e-8) for _ in range(num_tasks)]
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs, target_inputs=None):
        expert_outputs = self.mmoe_layer(inputs, target_inputs)
        return tuple(
            self.dropout(norm(tower(expert_outputs[i], target_inputs)))
            for i, (tower, norm) in enumerate(zip(self.tower_layer, self.normalizations))
        )
