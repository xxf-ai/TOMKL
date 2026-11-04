"""PyTorch implementation of the BASE attribute/behavior SASRec model.

Item and attribute IDs are 1..max_id, with zero reserved for left padding.
Behavior IDs are purchase=0, favorite/cart=2, view=3, and padding=4.
The optional extension letters retain the original meaning: a=attribute
fusion, b=behavior experts, t=attribute contrastive, c=expert contrastive.
The original local SASRec path always runs, regardless of the ``l`` letter.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .behavior_experts import TBRE


class _AttentionBlock(nn.Module):
    """Causal multihead attention and the original width-preserving FFN."""

    def __init__(self, dim, heads, dropout):
        super().__init__()
        if dim % heads:
            raise ValueError("hidden_units must be divisible by num_heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(dim, eps=1e-8)
        self.ffn_first = nn.Linear(dim, dim)
        self.ffn_second = nn.Linear(dim, dim)
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(dim, eps=1e-8)

    def forward(self, x, key_valid, query_valid=None):
        batch_size, length, _ = x.shape
        if query_valid is None:
            query_valid = key_valid
        def split(projected):
            return projected.reshape(batch_size, length, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = split(self.query(x)), split(self.key(x)), split(self.value(x))
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        causal = torch.ones(length, length, device=x.device, dtype=torch.bool).tril()
        allowed = causal[None, None, :, :] & key_valid[:, None, None, :]
        scores = scores.masked_fill(~allowed, -1e4)
        attention = F.softmax(scores, dim=-1)
        # Entirely empty behavior histories should not attend to padding.
        attention = attention * allowed.any(dim=-1, keepdim=True)
        attention = attention * query_valid[:, None, :, None]
        attention = self.attention_dropout(attention)
        context = torch.matmul(attention, v).transpose(1, 2).reshape(batch_size, length, self.dim)
        x = self.attention_norm(x + self.output_dropout(context))
        ffn = self.ffn_dropout(F.relu(self.ffn_first(x)))
        ffn = self.ffn_dropout(self.ffn_second(ffn))
        x = self.ffn_norm((x + ffn) * query_valid.unsqueeze(-1))
        return x * query_valid.unsqueeze(-1)


class _DynamicAttributeAttention(nn.Module):
    """Attend over category, seller (when present), and brand per timestep."""

    def __init__(self, dim, dropout):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.dim = dim

    def forward(self, behavior_emb, attribute_reps):
        # behavior_emb: [B,L,D], attribute_reps: [B,L,A,D]
        q = self.query(behavior_emb).unsqueeze(-2)
        k = self.key(attribute_reps)
        v = self.value(attribute_reps)
        scores = (q * k).sum(dim=-1) / math.sqrt(self.dim)
        weights = self.attention_dropout(F.softmax(scores, dim=-1))
        return self.output_dropout((weights.unsqueeze(-1) * v).sum(dim=-2))


class Model(nn.Module):
    """BASE recommender with one forward pass per training or scoring batch."""

    def __init__(self, num_user, num_item, num_cat, num_seller, num_brand, args):
        super().__init__()
        self.num_user = int(num_user)
        self.num_item = int(num_item)
        self.num_cat = int(num_cat)
        self.num_seller = int(num_seller)
        self.num_brand = int(num_brand)
        self.maxlen = int(args.maxlen)
        self.num_units = int(args.hidden_units)
        self.dropout_rate = float(args.dropout_rate)
        self.num_heads = int(args.num_heads)
        self.l2_reg = float(args.l2_reg)
        self.ext_modules = str(args.ext_modules)
        self.dataset = str(args.dataset)
        self.has_seller = self.dataset in ("Tmall", "JD")
        self.att_cl_rate = float(args.att_cl_rate)
        self.bsmmoe_cl_rate = float(args.bsmmoe_cl_rate)
        if "c" in self.ext_modules and "b" not in self.ext_modules:
            raise ValueError("The c expert contrastive extension requires b behavior experts")
        if self.has_seller and self.num_seller < 1:
            raise ValueError(f"{self.dataset} requires seller IDs")

        dim = self.num_units
        self.item_embedding = nn.Embedding(self.num_item + 1, dim, padding_idx=0)
        self.cat_embedding = nn.Embedding(self.num_cat + 1, dim, padding_idx=0)
        self.brand_embedding = nn.Embedding(self.num_brand + 1, dim, padding_idx=0)
        self.seller_embedding = (
            nn.Embedding(self.num_seller + 1, dim, padding_idx=0) if self.has_seller else None
        )
        self.position_embedding = nn.Embedding(self.maxlen, dim)
        self.behavior_embedding = nn.Embedding(max(int(args.num_behavior), 4) + 1, dim)
        for embedding in (self.item_embedding, self.cat_embedding, self.brand_embedding,
                          self.seller_embedding):
            if embedding is not None:
                with torch.no_grad():
                    nn.init.xavier_uniform_(embedding.weight[1:])
                    embedding.weight[0].zero_()

        self.dropout = nn.Dropout(self.dropout_rate)
        self.item_input_norm = nn.LayerNorm(dim, eps=1e-8)
        self.local_blocks = nn.ModuleList(
            [_AttentionBlock(dim, self.num_heads, self.dropout_rate)
             for _ in range(int(args.num_blocks))]
        )
        # The TensorFlow model builds these branches even when b is disabled;
        # retaining them only when needed avoids unused computation.
        if "b" in self.ext_modules:
            self.behavior_blocks = nn.ModuleList([
                nn.ModuleList(
                    [_AttentionBlock(dim, self.num_heads, self.dropout_rate)
                     for _ in range(int(args.num_blocks_behavior))]
                ) for _ in range(3)
            ])
            self.tbre = TBRE(
                self.maxlen, int(args.mmoe_hidden_units), int(args.num_expert_sh),
                int(args.num_expert_sp), 3, int(args.tower_hidden_units), dim,
                dropout_rate=self.dropout_rate, l2_reg=self.l2_reg,
            )

        if "a" in self.ext_modules:
            self.attribute_norms = nn.ModuleList([
                nn.LayerNorm(dim, eps=1e-8)
                for _ in range(3 if self.has_seller else 2)
            ])
            self.attribute_blocks = nn.ModuleList([
                _AttentionBlock(dim, 1, self.dropout_rate)
                for _ in range(len(self.attribute_norms))
            ])
            self.dynamic_attribute_attention = _DynamicAttributeAttention(dim, self.dropout_rate)

        if "t" in self.ext_modules:
            number_of_attrs = 3 if self.has_seller else 2
            self.attribute_gate = nn.Linear((number_of_attrs + 1) * dim, number_of_attrs)
            self.attribute_cl_projection = nn.Linear(dim, dim, bias=False)
            nn.init.trunc_normal_(
                self.attribute_cl_projection.weight,
                std=(2.0 / (3 * dim + 1)) ** 0.5,
            )
        if "c" in self.ext_modules:
            self.behavior_cl_projection = nn.Linear(dim, dim, bias=False)
            nn.init.trunc_normal_(
                self.behavior_cl_projection.weight,
                std=(2.0 / (3 * dim + 1)) ** 0.5,
            )

    def _scaled_embedding(self, embedding, ids):
        return embedding(ids.long()) * math.sqrt(self.num_units)

    def _input_embedding(self, embedding, ids, positions, valid, norm):
        values = self._scaled_embedding(embedding, ids) + positions
        return norm(self.dropout(values) * valid.unsqueeze(-1)) * valid.unsqueeze(-1)

    @staticmethod
    def _last_behavior_memory(values, mask):
        """At each step, read the most recent position having this behavior."""
        batch_size, length, dim = values.shape
        indices = torch.arange(1, length + 1, device=values.device).expand(batch_size, -1)
        latest = torch.cummax(torch.where(mask, indices, 0), dim=1).values
        padded = torch.cat([values.new_zeros(batch_size, 1, dim), values], dim=1)
        return padded.gather(1, latest.unsqueeze(-1).expand(-1, -1, dim))

    def _encode(self, batch, purchase_view=False, include_contrastive=True):
        input_ids = batch["input_id"].long()
        behavior_ids = batch["behavior_id"].long()
        if input_ids.ndim != 2 or behavior_ids.shape != input_ids.shape:
            raise ValueError("input_id and behavior_id must be matching [batch, length] tensors")
        batch_size, length = input_ids.shape
        if length > self.maxlen:
            raise ValueError(f"Sequence length {length} exceeds maxlen={self.maxlen}")
        valid = input_ids.ne(0)
        positions = self.position_embedding(
            torch.arange(self.maxlen - length, self.maxlen, device=input_ids.device)
        ).unsqueeze(0)
        item_embs = self._scaled_embedding(self.item_embedding, input_ids)
        inputs = self.item_input_norm(self.dropout(item_embs + positions) * valid.unsqueeze(-1))
        inputs = inputs * valid.unsqueeze(-1)
        local = inputs
        for block in self.local_blocks:
            local = block(local, valid)

        result = {"item_embs": item_embs, "local": local, "valid": valid}
        purchase = behavior_ids.eq(0) & valid
        favorite = behavior_ids.eq(2) & valid
        view = behavior_ids.eq(3) & valid
        result.update(purchase=purchase, favorite=favorite, view=view)

        attr_embs = []
        if "a" in self.ext_modules or (include_contrastive and "t" in self.ext_modules):
            for name, embedding in (("cat", self.cat_embedding),
                                    ("seller", self.seller_embedding),
                                    ("brand", self.brand_embedding)):
                if embedding is None:
                    continue
                ids = batch[f"{name}_id"].long()
                emb = self._scaled_embedding(embedding, ids)
                result[f"{name}_embs"] = emb
                result[f"{name}_valid"] = ids.ne(0)
                attr_embs.append((name, emb))

        if "a" in self.ext_modules:
            reps = []
            for index, (_, emb) in enumerate(attr_embs):
                attr_input = self.attribute_norms[index](
                    self.dropout(emb + positions) * valid.unsqueeze(-1)
                ) * valid.unsqueeze(-1)
                reps.append(self.attribute_blocks[index](attr_input, valid))
            behavior_embs = self.behavior_embedding(behavior_ids)
            attr_stack = torch.stack(reps, dim=-2)
            context = self.dynamic_attribute_attention(behavior_embs, attr_stack)
            result["attribute_context"] = context * valid.unsqueeze(-1)

        if "b" in self.ext_modules:
            behavior_masks = (purchase, favorite, view)
            branch_reps = []
            for branch_blocks, mask in zip(self.behavior_blocks, behavior_masks):
                branch = inputs * mask.unsqueeze(-1)
                for block in branch_blocks:
                    branch = block(branch, mask)
                branch_reps.append(branch)
            multiview = sum(branch_reps)
            memories = [self._last_behavior_memory(multiview, mask) for mask in behavior_masks]
            stacked = torch.cat(memories, dim=-1)
            outputs = self.tbre(stacked)
            result["behavior_outputs"] = outputs
            if purchase_view:
                behavior_context = outputs[0]
            else:
                target = batch["target_behavior"].long()
                behavior_context = sum(
                    output * target.eq(behavior_code).unsqueeze(-1)
                    for output, behavior_code in zip(outputs, (0, 2, 3))
                )
            result["behavior_context"] = behavior_context

        sequence = local
        if "a" in self.ext_modules:
            sequence = sequence + result["attribute_context"]
        if "b" in self.ext_modules:
            sequence = sequence + result["behavior_context"]
        result["sequence"] = sequence
        return result

    @staticmethod
    def _contrastive_loss(anchor, positive, mask):
        """Same-position in-batch negatives, as in the original CL objective."""
        batch_size, length, dim = anchor.shape
        if batch_size < 2 or not mask.any():
            return anchor.sum() * 0.0
        # The original fills invalid positions of each negative sequence with
        # randomly selected valid positions from that same sequence.
        probs = torch.where(mask.any(dim=1, keepdim=True), mask.float(),
                            torch.ones_like(mask, dtype=torch.float))
        sampled = torch.multinomial(probs, num_samples=length, replacement=True)
        fill = positive.gather(1, sampled.unsqueeze(-1).expand(-1, -1, dim))
        filled = torch.where(mask.unsqueeze(-1), positive, fill)
        logits = torch.einsum("bld,kld->blk", anchor, filled)
        log_probability = F.log_softmax(torch.sigmoid(logits), dim=-1)
        positive_index = torch.arange(batch_size, device=anchor.device)
        positive_log_probability = log_probability[positive_index, :, positive_index]
        return -(positive_log_probability * mask).sum() / batch_size

    def _attribute_contrastive(self, encoded):
        purchase = encoded["purchase"]
        auxiliary = encoded["favorite"] | encoded["view"]
        local = encoded["local"]
        names = ["cat", "seller", "brand"] if self.has_seller else ["cat", "brand"]
        attributes = [encoded[f"{name}_embs"] for name in names]
        contribution = F.softmax(self.attribute_gate(torch.cat(attributes + [local], dim=-1)), dim=-1)
        weighted = sum(
            contribution[..., index, None] * emb for index, emb in enumerate(attributes)
        )
        anchors = [local * purchase.unsqueeze(-1) for _ in attributes]
        anchors.append(local * auxiliary.unsqueeze(-1))
        positives = [emb * purchase.unsqueeze(-1) for emb in attributes]
        positives.append(weighted * auxiliary.unsqueeze(-1))
        attribute_masks = [purchase & encoded[f"{name}_valid"] for name in names]
        auxiliary_mask = auxiliary & torch.stack(
            [encoded[f"{name}_valid"] for name in names], dim=0
        ).any(dim=0)
        mask = torch.cat(attribute_masks + [auxiliary_mask], dim=1)
        anchor = self.attribute_cl_projection(torch.cat(anchors, dim=1))
        positive = self.attribute_cl_projection(torch.cat(positives, dim=1))
        return self._contrastive_loss(anchor, positive, mask)

    def _behavior_contrastive(self, encoded):
        purchase = encoded["purchase"]
        anchor = self.behavior_cl_projection(encoded["behavior_context"])
        positive = self.behavior_cl_projection(encoded["item_embs"])
        return self._contrastive_loss(anchor, positive, purchase)

    def _regularization(self):
        if self.l2_reg <= 0:
            return self.item_embedding.weight.new_zeros(())
        id_embeddings = (self.item_embedding, self.cat_embedding, self.brand_embedding,
                         self.seller_embedding)
        total = sum(embedding.weight[1:].square().sum() for embedding in id_embeddings
                    if embedding is not None)
        total += self.position_embedding.weight.square().sum()
        total += self.behavior_embedding.weight.square().sum()
        return self.l2_reg * total / 2

    def forward(self, batch):
        """Return total and component losses for a sampled training batch."""
        encoded = self._encode(batch)
        sequence = encoded["sequence"]
        pos_ids = batch["pos_id"].long()
        neg_ids = batch["neg_id"].long()
        positive = (sequence * self._scaled_embedding(self.item_embedding, pos_ids)).sum(dim=-1)
        negative = (sequence * self._scaled_embedding(self.item_embedding, neg_ids)).sum(dim=-1)
        valid_target = pos_ids.ne(0)
        count = valid_target.sum().clamp_min(1)
        ranking_loss = ((F.softplus(-positive) + F.softplus(negative)) * valid_target).sum() / count
        item_loss = ranking_loss + self._regularization()
        att_loss = (
            self.att_cl_rate * self._attribute_contrastive(encoded)
            if "t" in self.ext_modules else item_loss.new_zeros(())
        )
        bsm_loss = (
            self.bsmmoe_cl_rate * self._behavior_contrastive(encoded)
            if "c" in self.ext_modules else item_loss.new_zeros(())
        )
        return {"loss": item_loss + att_loss + bsm_loss,
                "item_loss": item_loss, "att_loss": att_loss, "bsm_loss": bsm_loss}

    def score_candidates(self, batch, candidate_ids):
        """Score candidate IDs [B,C] using only history and the purchase tower.

        Validation/test targets and their attributes are deliberately absent:
        all evaluation examples are purchase recommendations, so the purchase
        tower is selected explicitly. This also fixes the original TensorFlow
        evaluation path, which required an unfed target_behavior placeholder.
        """
        encoded = self._encode(batch, purchase_view=True, include_contrastive=False)
        if candidate_ids.ndim != 2 or candidate_ids.size(0) != encoded["sequence"].size(0):
            raise ValueError("candidate_ids must be [batch, candidates]")
        final = encoded["sequence"][:, -1, :]
        # Score the shared item table once; expanding it to [B, C, D] would
        # allocate hundreds of MB for the full-catalog evaluation batches.
        all_scores = final @ self.item_embedding.weight.T * math.sqrt(self.num_units)
        return all_scores.gather(1, candidate_ids.long())
