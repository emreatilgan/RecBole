# -*- coding: utf-8 -*-
# @Time   : 2025/11/03
# @Author : Kilo Code
# @Email  : research@example.com
#
# CAGE-CTR: Context-Adaptive Gated Experts for CTR (CPU-friendly)
#
# This model separates features into identity (stable user/item preferences)
# and context (situational conditions), computes base/context towers,
# mixes K lightweight experts via a context gate, and applies a context-adaptive
# temperature scaling head for better calibration across contexts.

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import ContextRecommender
from recbole.model.layers import MLPLayers
from recbole.model.init import xavier_normal_initialization


class CageCTR(ContextRecommender):
    r"""CAGE-CTR: Context-Adaptive Gated Experts for CTR.

    Key ideas:
    - Split features into identity vs. context based on config-provided field names.
    - Compute base (identity) and context towers.
    - Use a softmax gate (from context tower) to combine K tiny experts over joint hidden.
    - Temperature head T(ctx) calibrates logits per context: z' = z / T(ctx).

    Config keys (with safe defaults):
    - embedding_size (int): inherited (default: 10)
    - dropout_prob (float): default 0.1
    - mlp_hidden_size_base (list[int]): default [64, 64]
    - mlp_hidden_size_ctx (list[int]): default [64]
    - expert_hidden_size (list[int]): default [64]
    - expert_num (int): default 3
    - temp_cap_min (float): default 0.5
    - temp_cap_max (float): default 2.0
    - lambda_inv (float): gate entropy regularizer (default 1e-3)
    - lambda_cal (float): temperature variance regularizer (default 1e-4)
    - identity_fields (list[str]): names for identity fields (default: [USER_ID_FIELD, ITEM_ID_FIELD] if present)
    - context_fields (list[str]): optional explicit list; when absent, context = all - identity
    - numerical_features (list[str]): inherited, respected by base class

    Notes:
    - Ordered field layout follows concat_embed_input_fields:
      [token_seq..., token..., float_seq..., float...]
    """

    def __init__(self, config, dataset):
        super(CageCTR, self).__init__(config, dataset)

        # Hyperparameters with robust defaults
        def _cfg_get(name, default):
            try:
                return config[name]
            except Exception:
                # some Configs still support 'in' and get()
                try:
                    return config.get(name, default)
                except Exception:
                    return default

        self.dropout_prob: float = float(_cfg_get("dropout_prob", 0.1))
        self.mlp_hidden_size_base: List[int] = list(_cfg_get("mlp_hidden_size_base", [64, 64]))
        self.mlp_hidden_size_ctx: List[int] = list(_cfg_get("mlp_hidden_size_ctx", [64]))
        self.expert_hidden_size: List[int] = list(_cfg_get("expert_hidden_size", [64]))
        self.expert_num: int = int(_cfg_get("expert_num", 3))
        self.temp_cap_min: float = float(_cfg_get("temp_cap_min", 0.5))
        self.temp_cap_max: float = float(_cfg_get("temp_cap_max", 2.0))
        self.lambda_inv: float = float(_cfg_get("lambda_inv", 1e-3))
        self.lambda_cal: float = float(_cfg_get("lambda_cal", 1e-4))

        # Build ordered field name list to align with concat_embed_input_fields
        self.ordered_sparse_names: List[str] = []
        if len(self.token_seq_field_names) > 0:
            self.ordered_sparse_names += list(self.token_seq_field_names)
        if len(self.token_field_names) > 0:
            self.ordered_sparse_names += list(self.token_field_names)

        self.ordered_dense_names: List[str] = []
        if len(self.float_seq_field_names) > 0:
            self.ordered_dense_names += list(self.float_seq_field_names)
        if len(self.float_field_names) > 0:
            self.ordered_dense_names += list(self.float_field_names)

        self.ordered_all_names: List[str] = self.ordered_sparse_names + self.ordered_dense_names

        # Determine identity/context fields
        default_identity_fields: List[str] = []
        # Prefer USER_ID_FIELD and ITEM_ID_FIELD if present among token fields
        try:
            uid = config["USER_ID_FIELD"]
            iid = config["ITEM_ID_FIELD"]
            if uid in self.ordered_all_names:
                default_identity_fields.append(uid)
            if iid in self.ordered_all_names:
                default_identity_fields.append(iid)
        except Exception:
            pass

        identity_fields: List[str] = list(_cfg_get("identity_fields", default_identity_fields))
        context_fields: List[str] = list(_cfg_get("context_fields", []))

        # Fallback: if no identity fields identified, pick first two token fields as identity
        if len(identity_fields) == 0:
            pick = []
            for name in self.ordered_sparse_names:
                if name == self.LABEL:
                    continue
                pick.append(name)
                if len(pick) >= 2:
                    break
            identity_fields = pick

        identity_set = set(identity_fields)
        if len(context_fields) == 0:
            context_fields = [n for n in self.ordered_all_names if (n not in identity_set) and (n != self.LABEL)]
        context_set = set(context_fields)

        # Indices for selection at runtime
        self.identity_indices: List[int] = [i for i, n in enumerate(self.ordered_all_names) if n in identity_set]
        self.context_indices: List[int] = [i for i, n in enumerate(self.ordered_all_names) if n in context_set]

        if len(self.identity_indices) == 0:
            raise RuntimeError(
                "CageCTR: No identity fields resolved. Provide config['identity_fields'] or ensure USER_ID_FIELD/ITEM_ID_FIELD exist in dataset."
            )
        if len(self.context_indices) == 0:
            # Permit empty context by using a dummy zero vector later, but emit a warning in logs
            self.logger.warning("CageCTR: No context fields found; context tower will operate on zeros.")

        # Compute input dims (each field contributes embedding_size after base embedding)
        id_in_dim = len(self.identity_indices) * self.embedding_size
        ctx_in_dim = len(self.context_indices) * self.embedding_size if len(self.context_indices) > 0 else self.embedding_size  # dummy

        # Define towers
        base_sizes = [id_in_dim] + self.mlp_hidden_size_base if id_in_dim > 0 else [self.embedding_size] + self.mlp_hidden_size_base
        ctx_sizes = [ctx_in_dim] + self.mlp_hidden_size_ctx

        self.base_mlp = MLPLayers(base_sizes, dropout=self.dropout_prob, last_activation=True)
        self.ctx_mlp = MLPLayers(ctx_sizes, dropout=self.dropout_prob, last_activation=True)

        base_out = self.mlp_hidden_size_base[-1] if len(self.mlp_hidden_size_base) > 0 else (id_in_dim if id_in_dim > 0 else self.embedding_size)
        ctx_out = self.mlp_hidden_size_ctx[-1] if len(self.mlp_hidden_size_ctx) > 0 else ctx_in_dim

        # Gate and experts
        self.gate = nn.Linear(ctx_out, self.expert_num)
        joint_in = base_out + ctx_out

        self.experts = nn.ModuleList()
        if len(self.expert_hidden_size) > 0:
            self.experts.extend(
                nn.Sequential(
                    MLPLayers([joint_in] + self.expert_hidden_size, dropout=self.dropout_prob, last_activation=True),
                    nn.Linear(self.expert_hidden_size[-1], 1),
                )
                for _ in range(self.expert_num)
            )
        else:
            self.experts.extend(nn.Linear(joint_in, 1) for _ in range(self.expert_num))

        # Temperature head
        self.temp_head = nn.Linear(ctx_out, 1)

        # Loss/activation
        self.loss_fn_bce = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

        # Parameters initialization
        self.apply(xavier_normal_initialization)

    def _select_and_flat(self, all_emb: torch.Tensor, indices: List[int]) -> torch.Tensor:
        # all_emb: [B, F, D], indices -> [B, len(indices) * D]
        if len(indices) == 0:
            # Provide a deterministic zero feature if one side is empty
            B, _, D = all_emb.shape
            return all_emb.new_zeros((B, D))
        sel = all_emb.index_select(dim=1, index=all_emb.new_tensor(indices, dtype=torch.long))
        return sel.reshape(sel.size(0), -1)

    def _temperature(self, h_ctx: torch.Tensor) -> torch.Tensor:
        T = F.softplus(self.temp_head(h_ctx)) + 1e-6
        T = torch.clamp(T, min=self.temp_cap_min, max=self.temp_cap_max)
        return T

    def forward(self, interaction) -> torch.Tensor:
        # Embed all fields with base class util, shape [B, num_field, embed_dim]
        all_embeddings = self.concat_embed_input_fields(interaction)  # [B, F, D]
        B = all_embeddings.size(0)
        # Split identity vs context
        h_id_in = self._select_and_flat(all_embeddings, self.identity_indices)
        h_ctx_in = self._select_and_flat(all_embeddings, self.context_indices)

        # Towers
        h_base = self.base_mlp(h_id_in)
        h_ctx = self.ctx_mlp(h_ctx_in)

        # Gate
        gate_logits = self.gate(h_ctx)  # [B, K]
        gate = F.softmax(gate_logits, dim=-1)

        # Experts on joint hidden
        joint = torch.cat([h_base, h_ctx], dim=-1)
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(joint))  # [B, 1]
        experts_stacked = torch.cat(expert_outputs, dim=-1)  # [B, K]
        z = torch.sum(gate * experts_stacked, dim=-1, keepdim=True)  # [B, 1]

        # Temperature scaling
        T = self._temperature(h_ctx)  # [B, 1]
        y_logit = z / T
        return y_logit.squeeze(-1)

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL].float()
        logits = self.forward(interaction)
        bce = self.loss_fn_bce(logits, label)

        # Recompute small pieces required for regularizers (avoid full forward duplication)
        all_embeddings = self.concat_embed_input_fields(interaction)
        h_ctx_in = self._select_and_flat(all_embeddings, self.context_indices)
        h_ctx = self.ctx_mlp(h_ctx_in)
        gate = F.softmax(self.gate(h_ctx), dim=-1)
        T = self._temperature(h_ctx)

        # Gate entropy penalty (encourage diversity, less peaky gates)
        p = gate + 1e-12
        entropy_term = torch.sum(p * torch.log(p), dim=-1).mean()  # <= 0
        r_inv = self.lambda_inv * entropy_term

        # Temperature variance penalty to discourage extreme context scaling
        var_T = torch.var(T, unbiased=False)
        r_cal = self.lambda_cal * var_T

        return bce + r_inv + r_cal

    def predict(self, interaction):
        return self.sigmoid(self.forward(interaction))